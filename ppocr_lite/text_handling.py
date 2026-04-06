import difflib
from typing import Iterator

from ppocr_lite.structs import OCRResult, BBox


def arrange_text(words: list[OCRResult], line_cy_threshold: float = 0.2,
                 word_dist_threshold: float | None = 5.0) -> list[list[OCRResult]]:
    """
    Arrange a list of [OCRResult]s in lines of ordered OCRResults.

    Parameters
    ----------
    words:
        The words to arrange as a list of [OCRResult]s.
    line_cy_threshold:
        The maximum allowed vertical distance between word center points in one line, relative to line
        height.
    word_dist_threshold:
        The maximum horizontal distance between words to be considered part of the same line, relative
        to line height (as a proxy for font size).
    """

    if not words:
        return []

    words = sorted(words, key=lambda w: w.box.cy)

    lines = [[words[0]]]
    min_y = words[0].box.y
    max_y = words[0].box.y2

    for w in words[1:]:
        threshold = (max_y - min_y) * line_cy_threshold
        if abs(w.box.cy - lines[-1][0].box.cy) < threshold:
            lines[-1].append(w)
            min_y = min(min_y, w.box.y)
            max_y = max(max_y, w.box.y2)
        else:
            lines.append([w])
            min_y = w.box.y
            max_y = w.box.y2

    lines = [sorted(line, key=lambda w: w.box.x) for line in lines]

    if word_dist_threshold is not None and word_dist_threshold not in (float("inf"), -1):
        res = []
        for line in lines:
            start_idx = 0
            line_height = max(w.box.y2 for w in line) - min(w.box.y for w in line)
            word_threshold = line_height * word_dist_threshold

            last_x = line[0].box.x

            for j, w in enumerate(line):
                if w.box.x - last_x > word_threshold:
                    res.append(line[start_idx:j])
                    start_idx = j
                last_x = w.box.x2

            if start_idx == 0 or start_idx < len(line) - 1:
                res.append(line[start_idx:])

        lines = res

    return lines


def merge_phrase_boxes(text_lines: list[list[OCRResult]], phrase_tokens: list[str]) -> Iterator[OCRResult]:
    """
    Slide a variable-width window over the word list and return all bounding
    boxes where the concatenated word text matches the concatenated phrase
    (case-insensitive, space-agnostic).

    Handles OCR artifacts where spaces are dropped and multiple phrase tokens
    are fused into a single word (e.g. "helloworld" instead of "hello world").

    Newlines in [phrase_tokens] are not supported; all text to search for must
    be in one line.

    Requires input text to be in the format returned by [arrange_text()].
    """

    if not phrase_tokens:
        return ()

    phrase_concat = "".join(t.lower() for t in phrase_tokens)

    for line in text_lines:
        for i in range(len(line)):
            running = ""
            for j in range(i, len(line)):
                running += line[j].text.lower()

                if running == phrase_concat:
                    window = line[i: j + 1]
                    yield OCRResult(
                        text=' '.join(w.text for w in window),
                        box=BBox.surrounding(tuple(w.box for w in window)),
                        score=sum(w.score for w in window) / len(window)
                    )

                # Once we've consumed more characters than the phrase, no match
                # is possible starting at i – move on.q
                if len(running) > len(phrase_concat):
                    break

    return ()


def merge_phrase_boxes_fuzzy(text_lines: list[list[OCRResult]], phrase_tokens: list[str], cutoff: float = 0.9) -> Iterator[OCRResult]:
    """
    A fallback to [merge_phrase_boxes()] that allows partial / substring word matches so
    that OCR misreads or slight variations still match (e.g. 'Subrnit' -> 'Submit').

    Requires input text to be in the format returned by [arrange_text()].
    """

    n = len(phrase_tokens)
    if n == 0:
        return ()

    phrase_concat = ''.join(phrase_tokens).lower().replace(" ", "")
    matcher = difflib.SequenceMatcher(a=phrase_concat)


    for line in text_lines:
        line_texts = [w.text.lower().replace(" ", "") for w in line]
        line_text_concat = ''.join(line_texts)

        for j in range(0, max(len(line_text_concat) - len(phrase_concat), 1)):
            matcher.set_seq2(line_text_concat[j: j + len(phrase_concat)])

            if matcher.real_quick_ratio() >= cutoff and \
                    matcher.quick_ratio() >= cutoff and \
                    matcher.ratio() >= cutoff:
                start_idx = 0
                end_idx = 0
                for i, seg in enumerate(line_texts):
                    before = sum(len(s) for s in line_texts[:i])
                    after = before + len(seg)
                    if before < j < after:
                        start_idx = i
                    if before < j + len(phrase_concat) < end_idx:
                        end_idx = i
                window = line[start_idx: end_idx + 1]
                yield OCRResult(
                    text=' '.join(w.text for w in window),
                    box=BBox.surrounding(tuple(w.box for w in window)),
                    score=sum(w.score for w in window) / len(window)
                )

    return ()
