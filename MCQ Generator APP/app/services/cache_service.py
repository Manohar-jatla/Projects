from hashlib import md5

CACHE = {}

def _make_key(filename: str, difficulty: str, num_questions: int):
    key_raw = f"{filename}:{difficulty}:{num_questions}"
    return md5(key_raw.encode()).hexdigest()

def get_cached_result(filename, difficulty, num_questions):
    return CACHE.get(_make_key(filename, difficulty, num_questions))

def save_to_cache(filename, difficulty, num_questions, result):
    CACHE[_make_key(filename, difficulty, num_questions)] = result
