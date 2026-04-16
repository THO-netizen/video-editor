"""
content_strategist.py
Content strategy engine using Google Gemini (free tier).
"""

import json
import re
import google.generativeai as genai


SYSTEM_PROMPT = """You are an elite social-media content strategist who specialises in short-form video
(YouTube Shorts, TikTok, Instagram Reels). You write in the high-energy, fast-paced style
of creators like Artem Saykin — punchy sentences, bold hooks, no fluff.

When given a video transcript and the creator's profession, you will:
1. Identify the core message and most engaging moments.
2. Write a ready-to-record script for the NEXT video in a logical content series.
3. Provide a 4-week Content Loop Strategy with specific video ideas for each week.

Return ONLY a valid JSON object with NO extra text, markdown, or code fences:
{
  "hook": "<one-sentence attention-grabbing hook for the next video>",
  "next_video_script": "<full word-for-word script with [PAUSE] and [ZOOM] cues>",
  "content_loop": [
    {"week": 1, "theme": "...", "videos": ["...", "...", "..."]},
    {"week": 2, "theme": "...", "videos": ["...", "...", "..."]},
    {"week": 3, "theme": "...", "videos": ["...", "...", "..."]},
    {"week": 4, "theme": "...", "videos": ["...", "...", "..."]}
  ],
  "posting_times": "<best days and times to post>",
  "hashtags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}"""


def generate_strategy(
    transcript: str,
    profession: str,
    api_key: str,
    model: str = "gemini-1.5-flash",
) -> dict:
    genai.configure(api_key=api_key)

    gemini = genai.GenerativeModel(
        model,
        generation_config={"temperature": 0.85, "max_output_tokens": 2500},
    )

    prompt = f"""{SYSTEM_PROMPT}

CREATOR PROFESSION: {profession or "Content Creator"}

VIDEO TRANSCRIPT:
\"\"\"{transcript[:6000]}\"\"\"

Return the JSON strategy now."""

    response = gemini.generate_content(prompt)
    raw = response.text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}
