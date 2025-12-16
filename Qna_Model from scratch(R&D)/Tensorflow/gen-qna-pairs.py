import pandas as pd
import json
import random
import ast

df = pd.read_csv("data/tmdb_5000_movies.csv")
df = df.dropna(subset=["title", "overview", "genres", "release_date", "vote_average", "budget", "revenue", "runtime"])

def parse_json_column(data):
    try:
        return ast.literal_eval(data)
    except:
        return []

def get_genre_names(genre_str):
    genres = parse_json_column(genre_str)
    return [g['name'] for g in genres]

def get_production_companies(companies_str):
    companies = parse_json_column(companies_str)
    return [c['name'] for c in companies]

def generate_qna_pairs():
    qna = []

    for _, row in df.iterrows():
        title = row["title"]
        overview = row["overview"]
        genres = get_genre_names(row["genres"])
        release_year = row["release_date"][:4]
        vote = float(row["vote_average"])
        budget = int(row["budget"])
        revenue = int(row["revenue"])
        runtime = int(row["runtime"])
        prod_companies = get_production_companies(row.get("production_companies", "[]"))
        company = prod_companies[0] if prod_companies else "an unknown company"

        # Basic
        qna.append((f"What is {title} about?", overview))
        qna.append((f"When was {title} released?", f"{title} was released in {release_year}."))
        qna.append((f"What is the genre of {title}?", f"{title} falls under genres: {', '.join(genres)}."))
        qna.append((f"What is the runtime of {title}?", f"{title} runs for {runtime} minutes."))
        qna.append((f"Which company produced {title}?", f"{title} was produced by {company}."))

        # Rating-based
        if vote >= 8:
            qna.append((f"Is {title} a top-rated movie?", f"Yes, {title} is highly rated with {vote} rating."))
        elif vote <= 5:
            qna.append((f"Is {title} poorly rated?", f"Yes, {title} has a low rating of {vote}."))

        # Budget & Revenue
        if budget > 100_000_000:
            qna.append((f"Was {title} expensive to make?", f"Yes, the budget of {title} was over $100M."))
        if revenue > 300_000_000:
            qna.append((f"Did {title} perform well at the box office?", f"Yes, it earned over $300M in revenue."))

        # Comparisons
        qna.append((f"What is the difference between the budget and revenue of {title}?",
                    f"The difference is ${revenue - budget}."))

    # Global stats
    action_count = df['genres'].apply(lambda x: 'Action' in get_genre_names(x)).sum()
    comedy_count = df['genres'].apply(lambda x: 'Comedy' in get_genre_names(x)).sum()
    long_movies = df[df['runtime'] > 150]

    qna += [
        ("How many Action movies are there?", f"There are {action_count} Action movies."),
        ("How many Comedy movies are there?", f"There are {comedy_count} Comedy movies."),
        ("Which movie has the highest rating?", f"{df.loc[df['vote_average'].idxmax()]['title']} is the highest rated."),
        ("Which movie has the highest revenue?", f"{df.loc[df['revenue'].idxmax()]['title']} earned the highest revenue."),
        ("Which movie has the biggest budget?", f"{df.loc[df['budget'].idxmax()]['title']} had the highest budget."),
        ("List movies longer than 150 minutes.", 
         f"Some long movies are: {', '.join(long_movies['title'].head(5).tolist())}."),
        ("Name 3 movies released in 2015.",
         f"{', '.join(df[df['release_date'].str.startswith('2015')]['title'].head(3).tolist())}")
    ]

    random.shuffle(qna)
    print(f"Generated {len(qna)} QnA pairs.")

    with open("qna_pairs.json", "w", encoding="utf-8") as f:
        json.dump(qna, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    generate_qna_pairs()
