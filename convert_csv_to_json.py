import csv
import json
import argparse
from bs4 import BeautifulSoup


def normalize_text(t):
    if not t:
        return ""
    return t.replace("\u00a0", " ").strip()


def extract_html_info(html_text):

    soup = BeautifulSoup(html_text, "html.parser")

    sub_lemmas = [i.get_text(strip=True) for i in soup.find_all("i")]
    phrases = [b.get_text(strip=True) for b in soup.find_all("b")]

    # Remove ALL tags
    clean_definition = soup.get_text(" ", strip=True)

    return clean_definition, sub_lemmas, phrases


def csv_to_json_rows(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, start=1):
            lemma = normalize_text(row.get("Entry"))
            meaning_id = normalize_text(row.get("Poly-Index"))
            raw_definition = normalize_text(row.get("Content"))
            script = normalize_text(row.get("Script"))
            pronunciation = normalize_text(row.get("Pronunciation"))
            variants = normalize_text(row.get("Optional-Suffix"))

            definition, sub_lemmas, phrases = extract_html_info(raw_definition)

            entry = {
                "id": i,
                "lemma": lemma,
                "sense_id": meaning_id,
                "definition": definition,
                "sub_lemmas": sub_lemmas,
                "phrases": phrases,
                "variants": [v.strip() for v in variants.split("|") if v.strip()],
                "script": script,
                "pronunciation": pronunciation,
            }

            rows.append(entry)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    data = csv_to_json_rows(args.input)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(data)} entries → {args.output}")


if __name__ == "__main__":
    main()
