import nacl.exceptions
try:
    import oqs
except (RuntimeError, ImportError):
    oqs = None
import nacl
from nacl.signing import VerifyKey


class Verifier:
    def __init__(self, algorithm, key_path):
        with open(key_path, "rb") as f:
            self.key = f.read()
        if algorithm == "ed25519":
            self.verifier = VerifyKey(self.key)
        elif algorithm in oqs.get_enabled_sig_mechanisms():
            # OQS verifier doesn't take a key directly, it uses the public key
            self.verifier = oqs.Signature(algorithm)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def verify(self, message, signature):
        if isinstance(self.verifier, VerifyKey):
            try:
                result = self.verifier.verify(message, signature)
                return True
            except Exception as e:
                return False
        else:
            result = self.verifier.verify(message, signature, self.key)
            return result