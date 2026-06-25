#!/usr/bin/env python3
"""
MusicMind for Plex - Artist Metadata Enrichment
Enriches artist records with gender, country, era, and group type via OpenAI.
"""

import sqlite3
import json
import time
from openai import OpenAI
from datetime import datetime
from config import DB_PATH, OPENAI_KEY

BATCH_SIZE = 20

client = OpenAI(api_key=OPENAI_KEY)

def init_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS artist_meta (
            artist        TEXT PRIMARY KEY,
            gender        TEXT,
            country       TEXT,
            era           TEXT,
            group_type    TEXT,
            active_since  INTEGER,
            enriched_at   TEXT
        )
    """)
    conn.commit()
    print("Table ready.\n")

def get_unenriched_artists(conn):
    return [row[0] for row in conn.execute("""
        SELECT DISTINCT artist FROM tracks
        WHERE artist IS NOT NULL AND artist != ''
          AND artist NOT IN (SELECT artist FROM artist_meta)
        ORDER BY artist
    """).fetchall()]

def enrich_batch(batch):
    artist_list = "\n".join(f"{i+1}. {a}" for i, a in enumerate(batch))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[{
            "role": "user",
            "content": f"""For each musician/artist/band below, return metadata.

Fields:
- gender: "female", "male", "mixed" (band with multiple genders), "unknown"
- country: 2-letter ISO country code (e.g. "US", "UK", "BR", "FR") or "unknown"
- era: primary decade of activity — "50s", "60s", "70s", "80s", "90s", "00s", "10s", "20s", or "unknown"
- group_type: "solo", "duo", "band", "orchestra", "dj", "unknown"
- active_since: year as integer, or null if unknown

Respond ONLY with a JSON array, one object per artist in order.
Format: [{{"artist": "name", "gender": "...", "country": "...", "era": "...", "group_type": "...", "active_since": null}}, ...]

Artists:
{artist_list}"""
        }]
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)

def main():
    print("MusicMind for Plex - Artist Metadata Enrichment")
    print("=" * 50)

    conn = sqlite3.connect(DB_PATH, timeout=30)
    init_table(conn)

    artists = get_unenriched_artists(conn)
    total = len(artists)
    print(f"Artists to enrich: {total}\n")

    if total == 0:
        print("All artists already enriched!")
        conn.close()
        return

    done = 0
    now = datetime.now().isoformat()

    for i in range(0, total, BATCH_SIZE):
        batch = artists[i:i+BATCH_SIZE]
        try:
            results = enrich_batch(batch)
            for j, result in enumerate(results):
                if j >= len(batch):
                    break
                conn.execute("""
                    INSERT OR REPLACE INTO artist_meta
                        (artist, gender, country, era, group_type, active_since, enriched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    batch[j],
                    result.get('gender', 'unknown'),
                    result.get('country', 'unknown'),
                    result.get('era', 'unknown'),
                    result.get('group_type', 'unknown'),
                    result.get('active_since'),
                    now
                ))
            conn.commit()
            done += len(batch)
            print(f"  {done}/{total} artists enriched")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Batch failed: {e}")
            time.sleep(2)

    print(f"\nDone. Enriched {done} artists.")

    # Quick summary
    print("\n=== Summary ===")
    for label, field, val in [
        ("Female artists",  "gender",     "female"),
        ("Male artists",    "gender",     "male"),
        ("Mixed bands",     "gender",     "mixed"),
        ("US artists",      "country",    "US"),
        ("UK artists",      "country",    "UK"),
        ("Solo artists",    "group_type", "solo"),
        ("Bands",           "group_type", "band"),
    ]:
        count = conn.execute(
            f"SELECT COUNT(*) FROM artist_meta WHERE {field} = ?", (val,)
        ).fetchone()[0]
        print(f"  {label:20s}  {count}")

    conn.close()

if __name__ == "__main__":
    main()
