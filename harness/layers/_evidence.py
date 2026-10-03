"""Bằng chứng dùng chung cho `critic` và `citation_checker`.

Cả hai lớp hỏi cùng một câu: "câu này là trích dẫn của tài liệu nào mà
lượt chạy ĐÃ THỰC SỰ NHÌN THẤY?". Trả lời ở một chỗ để hai lớp không bao
giờ bất đồng với nhau.

So khớp theo đúng cách bộ chấm so (`arena.scorer._norm` / `_supports`):
NFC + casefold + gộp khoảng trắng, và trong phạm vi MỘT DÒNG của tài liệu.
Chuẩn hoá chỉ dùng để SO SÁNH — không lớp nào ghi chuỗi đã chuẩn hoá trở
lại vào `claim["text"]`.
"""

from __future__ import annotations

import re
import unicodedata

#: Ngắn hơn ngưỡng này thì bộ chấm không coi là trích dẫn của bất kỳ tài
#: liệu nào (`arena.scorer.MIN_SUPPORT_CHARS`).
MIN_QUOTE_CHARS = 12

_WS_RE = re.compile(r"\s+")


def norm(text) -> str:
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def _lines(body) -> list[str]:
    if not isinstance(body, str):
        return []
    return [line for line in (norm(raw) for raw in body.splitlines()) if line]


def quotes_a_line(text, body) -> bool:
    """`text` có phải trích dẫn nguyên văn của MỘT dòng trong `body` không."""
    wanted = norm(text)
    if len(wanted) < MIN_QUOTE_CHARS:
        return False
    return any(wanted in line for line in _lines(body))


def seen_docs(ctx) -> list:
    """Tài liệu mà lượt chạy đã quan sát: về nguyên vẹn từ một lần fetch,
    hoặc có mã xuất hiện trong kết quả search. Trích một tài liệu ngoài
    danh sách này bị chấm `UNRETRIEVED`."""
    corpus = getattr(ctx, "corpus", None)
    if corpus is None:
        return []
    observed = ctx.observed_text
    return [
        doc
        for doc in corpus.docs
        if doc.doc_id in observed or (doc.body and doc.body in observed)
    ]


def sources(ctx, text, docs=None) -> list:
    """Các tài liệu ĐÃ QUAN SÁT có một dòng chứa nguyên văn `text`."""
    if docs is None:
        docs = seen_docs(ctx)
    return [doc for doc in docs if quotes_a_line(text, doc.body)]
