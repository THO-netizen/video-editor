"""
content_strategist.py
GPT-4 powered content strategy engine.
Analyses the transcript + user's profession to generate:
  - A full script for the NEXT video in a series
  - A 4-week content loop strategy
"""

import openai
from typing import Dict


SYSTEM_PROMPT = """\
You are an elite social-media content strategist who specialises in short-form video \
(YouTube Shorts, TikTok, Instagram Reels). You write in the high-energy, fast-paced style \
of creators like Artem Saykin — punchy sentences, bold hooks, no fluff.

When given a video transcript and the creator's profession, you will:
1. Identify the core message and most engaging moments.
2. Write a ready-to-record script for the NEXT video in a logical content series.
3. Provide a 4-week Content Loop Strategy with specific video ideas for each week.

Format your output as JSON with the following keys:
{
  "hook": "<one-sentence attention-grabbing hook for the next video>",
  "next_video_script": "<full word-for-word script, formatted with [PAUSE] markers and [ZOOM] cues>",
  "content_loop": [
    {"week": 1, "theme": "...", "videos": ["...", "...", "..."]},
    {"week": 2, "theme": "...", "videos": ["...", "...", "..."]},
    {"week": 3, "theme": "...", "videos": ["...", "...", "..."]},
    {"week": 4, "theme": "...", "videos": ["...", "...", "..."]}
  ],
  "posting_times": "<best days and times to post based on the niche>",
  "hashtags": ["tag1", "tag2", "..."]
}
"""


def generate_strategy(
    transcript: str,
    profession: str,
    api_key: str,
    model: str = "gpt-4o",
) -> Dict:
    """
    Call GPT-4o to generate a content strategy.
    Returns parsed JSON dict.
    """
    client = openai.OpenAI(api_key=api_key)

    user_message = f"""
CREATOR PROFESSION: {profession or "Content Creator"}

VIDEO TRANSCRIPT:
\"\"\"
{transcript[:6000]}
\"\"\"

Please analyse this transcript and generate the full strategy JSON.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.85,
        max_tokens=2500,
    )

    import json
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}
