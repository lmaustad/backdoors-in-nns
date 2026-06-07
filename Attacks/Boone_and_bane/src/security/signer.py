try:
    import oqs
except (RuntimeError, ImportError):
    oqs = None
from nacl.signing import SigningKey


class Signer:
    def __init__(self, algorithm, secret_key_path=None):
        if secret_key_path:
            with open(secret_key_path, "rb") as f:
                secret_key = f.read()
        else:
            secret_key = None
        if algorithm == "ed25519":
            self.signer = SigningKey(secret_key) if secret_key else SigningKey.generate()
        elif algorithm in oqs.get_enabled_sig_mechanisms():
            self.signer = oqs.Signature(algorithm, secret_key)
            if secret_key is None:
                self.signer.generate_keypair()

    def sign(self, message):
        if isinstance(self.signer, SigningKey):
            return self.signer.sign(message).signature
        else:
            return self.signer.sign(message)