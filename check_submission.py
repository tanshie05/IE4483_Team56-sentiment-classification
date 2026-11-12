"""IE4483 Project 1: validates a submission file against the lecturer's template.

Run: python check_submission.py --submission submission.csv [--test-json test.json] [--template TEMPLATE.csv]

The lecturer's template is test.json written as "reviews,predictions" with a placeholder 0 in every row (minimal
quoting, CRLF after every row); without --template it is rebuilt from test.json, which reproduces it byte for byte.

The file is parsed as CSV; line counts are meaningless because some reviews contain quoted line breaks. Checks:
UTF-8 text; header exactly "reviews,predictions"; every row has exactly two fields (no index column); 1,851 rows;
reviews identical to the template, in the same order, quoted line breaks included (and to test.json when given);
predictions are the integers 0 or 1; every row ends in CRLF. Exits with status 1 if any check fails.
"""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

EXPECTED_HEADER = ['reviews', 'predictions']
EXPECTED_ROWS = 1851


def read_csv_rows(path):
    raw = Path(path).read_bytes()
    text = raw.decode('utf-8')  # Raises if the file is not UTF-8.
    return raw, text, list(csv.reader(io.StringIO(text, newline='')))


def record_terminators(text):
    """Quote-aware scan: for each line break outside quotes, whether it is CRLF. Also counts bare CRs outside quotes."""
    in_quotes, terminators, bare_cr = False, [], 0
    for i, char in enumerate(text):
        if char == '"':
            in_quotes = not in_quotes  # An escaped "" toggles twice, so it has no net effect.
        elif not in_quotes and char == '\n':
            terminators.append(i > 0 and text[i - 1] == '\r')
        elif not in_quotes and char == '\r' and (i + 1 >= len(text) or text[i + 1] != '\n'):
            bare_cr += 1
    return terminators, bare_cr, in_quotes


def template_text_from_test(test_json='test.json'):
    """The lecturer's template, rebuilt from test.json: header, then every review with a placeholder prediction 0."""
    reviews = [str(item['reviews']) for item in json.loads(Path(test_json).read_text(encoding='utf-8'))]
    return write_rows([EXPECTED_HEADER] + [[review, '0'] for review in reviews])


def read_template(template=None, test_json='test.json'):
    """(text, rows) of the template file if given, otherwise of the template rebuilt from test.json."""
    if template:
        _, text, rows = read_csv_rows(template)
    else:
        text = template_text_from_test(test_json)
        rows = list(csv.reader(io.StringIO(text, newline='')))
    return text, rows


def write_rows(rows):
    """Serialises rows exactly as the lecturer's template is written: minimal quoting, CRLF after every row."""
    buffer = io.StringIO(newline='')
    csv.writer(buffer, lineterminator='\r\n', quoting=csv.QUOTE_MINIMAL).writerows(rows)
    return buffer.getvalue()


def check(submission, template, test_json=None):
    results = []

    def record(name, ok, detail=''):
        results.append((name, bool(ok), detail))

    try:
        raw, text, rows = read_csv_rows(submission)
    except UnicodeDecodeError as error:
        record('UTF-8 text', False, str(error))
        return results
    record('UTF-8 text, no byte-order mark', not text.startswith('﻿'))
    header, body = (rows[0] if rows else []), rows[1:]
    record('Header is "reviews,predictions"', header == EXPECTED_HEADER, f'found {header}')
    widths = sorted({len(row) for row in rows})
    record('Every row has exactly 2 fields (no index column)', widths == [2], f'field counts found: {widths}')
    record(f'{EXPECTED_ROWS:,} data rows', len(body) == EXPECTED_ROWS, f'found {len(body):,}')

    template_text, template_rows = read_template(template, test_json or 'test.json')
    template_reviews = [row[0] for row in template_rows[1:]]
    reviews = [row[0] if row else None for row in body]
    mismatch = next((i for i, (a, b) in enumerate(zip(reviews, template_reviews)) if a != b), None)
    record('Reviews identical to the template and in the same order (quoted line breaks included)',
           len(reviews) == len(template_reviews) and mismatch is None,
           '' if mismatch is None else f'first difference at data row {mismatch + 1}')
    record('Quoted line breaks inside reviews preserved', sum(r.count('\n') for r in reviews if r) == sum(r.count('\n') for r in template_reviews),
           f"{sum(r.count(chr(10)) for r in reviews if r)} in the submission, {sum(r.count(chr(10)) for r in template_reviews)} in the template")
    if test_json:
        test_reviews = [str(item['reviews']) for item in json.loads(Path(test_json).read_text(encoding='utf-8'))]
        record('Reviews identical to test.json and in its order', reviews == test_reviews)

    predictions = [row[1] if len(row) > 1 else None for row in body]
    bad = [i + 1 for i, p in enumerate(predictions) if p not in ('0', '1')]
    record('Predictions are the integers 0 or 1', not bad, f'{len(bad)} bad values, first at data row {bad[0]}' if bad else '')

    terminators, bare_cr, unclosed = record_terminators(text)
    record('Every row ends in CRLF', len(terminators) == len(rows) and all(terminators) and bare_cr == 0 and not unclosed and text.endswith('\r\n'),
           f'{sum(terminators)} CRLF and {len(terminators) - sum(terminators)} LF row endings for {len(rows)} rows, {bare_cr} bare CR, '
           f'unclosed quote: {unclosed}')
    same_layout = write_rows([header] + [[r, p] for r, p in zip(reviews, predictions)]) == text
    record('Same layout as the template apart from predictions (minimal quoting)', same_layout,
           '' if same_layout else "the file is not written with the template's quoting and line endings")
    if not bad and predictions:
        positives = predictions.count('1')
        results.append(('Predicted class balance (information)', True,
                        f'{positives:,} positive ({100 * positives / len(predictions):.1f}%), {len(predictions) - positives:,} negative'))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--submission', default='submission.csv')
    parser.add_argument('--template', default=None, help="The lecturer's template; default: rebuilt from --test-json")
    parser.add_argument('--test-json', default='test.json')
    args = parser.parse_args()
    results = check(args.submission, args.template, args.test_json)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'  (' + detail + ')' if detail else ''}")
    failed = [name for name, ok, _ in results if not ok]
    print(f'\n{args.submission}: ' + ('ALL CHECKS PASSED' if not failed else f'{len(failed)} CHECK(S) FAILED'))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
