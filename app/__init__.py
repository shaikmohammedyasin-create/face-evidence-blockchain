# app/__init__.py
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

