"""Shared fact-field contract for prompt construction and extraction validation.

The application-owned safety and decision fields are intentionally excluded.
"""
from __future__ import annotations

FIELD_SCHEMA = {
    "narrative": {"type": "text", "description": "verbatim incident narrative"},
    "threat": {"type": "binary", "description": "reported threat"},
    "impersonation": {"type": "binary", "description": "explicit impersonation claim"},
    "platform": {"type": "text", "description": "digital platform"},
    "details": {"type": "text", "description": "incident details"},
    "publication_threat": {"type": "binary", "description": "threat to publish"},
    "active_unauthorized_funds": {"type": "binary", "description": "currently unauthorized funds activity"},
    "fraud_description": {"type": "binary", "description": "explicit financial deception"},
    "initial_time": {"type": "time", "description": "incident start or first contact time"},
    "discovery_time": {"type": "time", "description": "reporter's discovery time"},
    "account_creation_time": {"type": "time", "description": "account creation time"},
    "latest_time": {"type": "time", "description": "latest contact or repeated-harm time"},
    "repeated": {"type": "binary", "description": "repeated threat or harassment, not repeated messages alone"},
    "requested_amount": {"type": "amount", "description": "amount demanded"},
    "paid_amount": {"type": "amount", "description": "amount actually paid"},
    "requested_amount_relation": {"type": "relation", "description": "same, additional, or correction of requested amount"},
    "paid_amount_relation": {"type": "relation", "description": "same, additional, or correction of paid amount"},
    "payment_method": {"type": "text", "description": "payment method"},
    "deadline": {"type": "text", "description": "specified payment deadline"},
    "published": {"type": "binary", "description": "whether publication actually occurred"},
    "identity": {"type": "text", "description": "suspected actor identity, not reporter identity"},
    "contact": {"type": "text", "description": "availability or disclosed identifier of contact channel"},
    "contact_identifier_disclosure": {"type": "text", "description": "whether the reporter shared or declined to share the identifier here"},
    "evidence": {"type": "text", "description": "evidence the reporter says is available; never imply the app received or reviewed it"},
    "account_access": {"type": "binary", "description": "unauthorized access to the reporter's original account"},
    "original_account_access_available": {"type": "binary", "description": "reporter says they can still access their original account"},
    "harassment": {"type": "binary", "description": "explicit harassment"},
    "contact_repeated": {"type": "binary", "description": "repeated contact without assuming repeated harm"},
}

BINARY_FIELDS = {name for name, spec in FIELD_SCHEMA.items() if spec["type"] == "binary"}
ALLOWED_EXTRACTION_FIELDS = set(FIELD_SCHEMA)
ALLOWED_STATUSES = {"reported", "denied", "unknown", "not_remembered", "declined"}
ENGINE_OWNED_FIELDS = {"direct_danger", "priority_score", "assessment", "confirmed", "stage", "summary_version"}
RELATION_VALUES = {
    "requested_amount_relation": {"same", "additional", "correction"},
    "paid_amount_relation": {"same", "additional", "correction"},
}

FIELD_ANCHORS = {
    "threat": ("تهديد", "هدد", "يهدد", "ابتزاز", "يبتز", "يلاحق", "لاحق", "يؤذ", "اذي", "يضر", "اضر", "ضرر"),
    "impersonation": ("انتحال", "منتحل", "ينتحل", "تنتحل دوري", "تدعي انها انا", "يدعي انه انا", "باسمي", "يتظاهر انه انا", "كأنه انا", "كأنها انا"),
    "publication_threat": ("ينشر", "نشر", "ينزل", "نزل", "بفضح", "يفضح", "يهدد"),
    "repeated": ("تكرر", "كرر", "تكرار", "مستمر", "مرة ثانية", "مره ثانيه", "رسالة ثانية", "رساله ثانيه", "رجع بعت", "رجع ارسل", "رجع هدد"),
    "published": ("نشر", "نزل", "انتشرت", "بعت الصور", "حاط"),
    "active_unauthorized_funds": ("يسحب", "تنسحب", "بدون اذني", "دون اذني", "غير ماذون"),
    "fraud_description": ("نصب", "احتيال", "خدعني", "احتيال مالي"),
    "account_access": ("اخترق", "دخل على حساب", "دخل حساب", "سرق حساب"),
    "original_account_access_available": ("بقدر ادخل", "استطيع الدخول", "ما زلت اقدر ادخل", "حسابي معي"),
    "harassment": ("مضايقة", "تحرش", "يلاحق", "بلاحق", "بضايق"),
    "contact_repeated": ("تكرر", "كرر", "رسالة ثانية", "رساله ثانيه", "رجع بعت", "عدة رسائل", "مرات"),
    "discovery_time": ("اكتشفت", "اكتشف", "انتبهت", "لاحظت"),
    "account_creation_time": ("انشا", "انشاء", "انشئ", "انعمل", "موعد انشائه", "تاريخ انشائه"),
    "initial_time": ("بدا", "بداية", "اول تواصل", "اول رسالة", "بدأت الواقعة"),
    "latest_time": ("رجع", "اخر تواصل", "اخر مرة", "مرة ثانية"),
    "requested_amount": ("طلب", "طالب", "المبلغ المطلوب", "دينار", "ريال", "دولار"),
    "paid_amount": ("دفعت", "حولت", "تحويل", "دينار", "ريال", "دولار"),
    "requested_amount_relation": ("تصحيح", "بدل", "نفس المبلغ", "اضافي", "إضافي"),
    "paid_amount_relation": ("مش دفعة جديدة", "نفس الدفعة", "دفعة ثانية", "تحويل ثاني", "تصحيح"),
    "payment_method": ("محفظة", "بطاقة", "بنك", "تحويل", "نقد"),
    "deadline": ("مهلة", "موعد", "قبل", "وقت للدفع"),
    "identity": ("الفاعل", "هويته", "هويتها", "مجهول", "ما بعرف مين", "مين هو", "اسمي", "صورتي", "حسابي"),
    "contact": ("رقم", "حساب", "اسم المستخدم", "واتساب", "تواصل"),
    "contact_identifier_disclosure": ("ما بدي اكتب", "ما بدي اذكر", "لن اكتب", "ما كتبت"),
    "evidence": ("دليل", "نسخة احتياطية", "صور", "لقطات", "رسائل", "محادثات", "ايصال"),
    "narrative": (),
    "platform": ("واتساب", "انستغرام", "انستجرام", "فيسبوك", "سناب", "تيك توك"),
    "details": ("قال", "رسالة", "تصرف", "حساب", "صور", "تهديد"),
}
