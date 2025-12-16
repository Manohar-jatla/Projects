import os
import re
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# ------------------------------------------------------
# Device Setup
# ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# ------------------------------------------------------
# Tokenizer
# ------------------------------------------------------
def tokenize(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()

# ------------------------------------------------------
# Vectorization and Padding
# ------------------------------------------------------
def vectorize(text, vocab, add_sos_eos=True):
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if add_sos_eos:
        ids = [vocab["<SOS>"]] + ids + [vocab["<EOS>"]]
    return torch.tensor(ids, dtype=torch.long)

def pad(tensors, vocab):
    if len(tensors) == 0:
        return torch.empty((0,))
    max_len = max(len(t) for t in tensors)
    padded = []
    for t in tensors:
        if len(t) < max_len:
            padding = torch.full((max_len - len(t),), vocab["<PAD>"], dtype=torch.long)
            t = torch.cat([t, padding])
        padded.append(t)
    return torch.stack(padded)

# ------------------------------------------------------
# Model Classes
# ------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        emb = self.embed(x)
        _, h = self.rnn(emb)
        return h

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h):
        emb = self.embed(x.unsqueeze(1))
        out, h = self.rnn(emb, h)
        out = self.out(out.squeeze(1))
        return out, h

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        h = self.encoder(src)
        outputs = []
        input_token = tgt[:, 0]
        for t in range(1, tgt.shape[1]):
            out, h = self.decoder(input_token, h)
            outputs.append(out)
            input_token = tgt[:, t]
        return torch.stack(outputs, dim=1)

# ------------------------------------------------------
# Embedding-based QA
# ------------------------------------------------------
def build_qa_embeddings(qa_pairs, embedder):
    questions = [q for q, a in qa_pairs]
    embeddings = embedder.encode(
        questions, 
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return embeddings

def find_exact_answer(question, qa_pairs, qa_embeddings, embedder, threshold=0.6):
    if not question.strip():
        return None
    question_emb = embedder.encode(
        question, 
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    similarities = util.cos_sim(question_emb, qa_embeddings)[0]
    best_score, idx = torch.max(similarities, dim=0)
    if best_score >= threshold:
        return qa_pairs[idx][1]
    return None

# ------------------------------------------------------
# Lookup-based fallback
# ------------------------------------------------------
def answer_from_table(question, df):
    id_match = re.search(r"game\s+([A-Za-z0-9]+)", question.lower())
    if not id_match:
        return None
    game_id = id_match.group(1)
    for col in df.columns:
        if col.lower() in question.lower():
            matched_col = col
            break
    else:
        return None
    row = df.loc[df["id"] == game_id]
    if row.empty:
        return f"I couldn't find game ID {game_id}."
    value = row.iloc[0][matched_col]
    return f"the {matched_col} of game {game_id} is {value}."

# ------------------------------------------------------
# Inference
# ------------------------------------------------------
def infer(question, model, vocab, inv_vocab, df, qa_pairs, qa_embeddings, embedder):
    if not question.strip():
        return "(empty question)"

    # 1. Check embedding-based Q&A
    direct = find_exact_answer(question, qa_pairs, qa_embeddings, embedder)
    if direct:
        return direct

    # 2. Check table lookup
    table_answer = answer_from_table(question, df)
    if table_answer:
        return table_answer

    # 3. Run seq2seq model
    model.eval()
    with torch.no_grad():
        vec = vectorize(question, vocab).unsqueeze(0).to(device)
        h = model.encoder(vec)
        input_token = torch.tensor([vocab["<SOS>"]], device=device)
        result = []
        for _ in range(30):
            out, h = model.decoder(input_token, h)
            token = out.argmax(1).item()
            if token == vocab["<EOS>"]:
                break
            result.append(token)
            input_token = torch.tensor([token], device=device)
    if not result:
        return "(no output generated)"
    return " ".join(inv_vocab.get(t, "<UNK>") for t in result)

# ------------------------------------------------------
# Generate QA pairs
# ------------------------------------------------------
def generate_qa_pairs(df):
    templates = [
        "what is the {col} of game {game_id}?",
        "tell me game {game_id}'s {col}.",
        "which {col} does game {game_id} have?",
        "give me the {col} of game {game_id}.",
        "can you tell me the {col} of game {game_id}?",
        "who won game {game_id}?",
        "what opening was played in game {game_id}?",
    ]

    a_templates = [
        "the {col} of game {game_id} is {value}.",
        "game {game_id}'s {col} is {value}.",
        "{value} is the {col} of game {game_id}.",
        "for game {game_id}, the {col} is {value}.",
        "it is {value} for the {col} of game {game_id}.",
    ]

    qa_pairs = []

    for _, row in df.iterrows():
        game_id = str(row["id"])
        for col in df.columns:
            if col.lower() != "id":
                for t in templates:
                    q = t.format(col=col.lower(), game_id=game_id)
                    a = random.choice(a_templates).format(
                        col=col.lower(), game_id=game_id, value=row[col]
                    )
                    qa_pairs.append((q, a))

    # Add global filters
    if "winner" in df.columns:
        white_wins = (df["winner"] == "white").sum()
        black_wins = (df["winner"] == "black").sum()
        qa_pairs.append(("how many games did white win?", f"white won {white_wins} games."))
        qa_pairs.append(("how many games did black win?", f"black won {black_wins} games."))

    if "opening_name" in df.columns:
        sicilian = df["opening_name"].str.contains("Sicilian", case=False, na=False).sum()
        qa_pairs.append(("how many games used the Sicilian Defense?", f"{sicilian} games used the Sicilian Defense."))

    if "victory_status" in df.columns:
        resign_count = (df["victory_status"] == "resign").sum()
        mate_count = (df["victory_status"] == "mate").sum()
        qa_pairs.append(("how many games ended with resignation?", f"{resign_count} games ended with resignation."))
        qa_pairs.append(("how many games ended with checkmate?", f"{mate_count} games ended with checkmate."))

    if "rated" in df.columns:
        rated_count = (df["rated"] == True).sum()
        unrated_count = (df["rated"] == False).sum()
        qa_pairs.append(("how many rated games are there?", f"there are {rated_count} rated games."))
        qa_pairs.append(("how many unrated games are there?", f"there are {unrated_count} unrated games."))

    if "turns" in df.columns:
        long_games = (df["turns"] > 50).sum()
        qa_pairs.append(("how many games had more than 50 moves?", f"{long_games} games had more than 50 moves."))

    qa_pairs.append(("how many games are there?", f"there are {len(df)} games."))
    print(f"✅ QA pairs generated: {len(qa_pairs)}")
    return qa_pairs

# ------------------------------------------------------
# Load or Train Model
# ------------------------------------------------------
random.seed(42)

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

if os.path.exists("qa_model.pt") and os.path.exists("vocab.pkl"):
    print("✅ Found saved model. Loading...")

    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    inv_vocab = {i: w for w, i in vocab.items()}

    encoder = Encoder(len(vocab), 64, 128).to(device)
    decoder = Decoder(len(vocab), 64, 128).to(device)
    model = Seq2Seq(encoder, decoder).to(device)
    model.load_state_dict(torch.load("qa_model.pt", map_location=device))
    model.eval()

    if os.path.exists("qa_pairs.pkl") and os.path.exists("qa_embeddings.pt"):
        with open("qa_pairs.pkl", "rb") as f:
            qa_pairs = pickle.load(f)
        qa_embeddings = torch.load("qa_embeddings.pt", map_location=device)
    else:
        print("⚠️ Embeddings not found. Rebuilding...")
        df = pd.read_csv("games.csv")
        qa_pairs = generate_qa_pairs(df)
        qa_embeddings = build_qa_embeddings(qa_pairs, embedder)
        with open("qa_pairs.pkl", "wb") as f:
            pickle.dump(qa_pairs, f)
        torch.save(qa_embeddings, "qa_embeddings.pt")

    csv_path = input("Enter path to CSV (e.g. games.csv): ").strip()
    df = pd.read_csv(csv_path)

else:
    csv_path = input("Enter path to CSV (e.g. games.csv): ").strip()
    df = pd.read_csv(csv_path)
    df = df.head(100)
    print("CSV Columns:", list(df.columns))

    qa_pairs = generate_qa_pairs(df)

    counter = Counter()
    for q, a in qa_pairs:
        counter.update(tokenize(q))
        counter.update(tokenize(a))

    vocab = {w: i + 4 for i, (w, _) in enumerate(counter.most_common())}
    vocab["<PAD>"] = 0
    vocab["<SOS>"] = 1
    vocab["<EOS>"] = 2
    vocab["<UNK>"] = 3
    inv_vocab = {i: w for w, i in vocab.items()}

    with open("qa_pairs.pkl", "wb") as f:
        pickle.dump(qa_pairs, f)

    vectorized_data = [(vectorize(q, vocab), vectorize(a, vocab)) for q, a in qa_pairs]
    random.shuffle(vectorized_data)
    questions, answers = zip(*vectorized_data)
    X = pad(questions, vocab).to(device)
    y = pad(answers, vocab).to(device)

    encoder = Encoder(len(vocab), 64, 128).to(device)
    decoder = Decoder(len(vocab), 64, 128).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in tqdm(range(50), desc="Training"):
        model.train()
        output = model(X, y)
        loss = loss_fn(output.view(-1, len(vocab)), y[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tqdm.write(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), "qa_model.pt")
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    qa_embeddings = build_qa_embeddings(qa_pairs, embedder)
    torch.save(qa_embeddings, "qa_embeddings.pt")
    print("✅ Model, vocab, and embeddings saved.")

# ------------------------------------------------------
# Interactive QA Loop
# ------------------------------------------------------
while True:
    q = input("\nAsk a question (or type 'exit'): ").strip()
    if q.lower() == "exit":
        break
    answer = infer(q, model, vocab, inv_vocab, df, qa_pairs, qa_embeddings, embedder)
    print("Answer:", answer)
