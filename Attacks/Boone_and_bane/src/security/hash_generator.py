import hmac

class HashGenerator:
    def __init__(self, key_path):
        with open(key_path, 'rb') as f:
            self.secret_key = f.read()

    def generate_hash(self, message):
        """
        Generate a hash for the given message using HMAC with SHA-256.
        """
        if not isinstance(message, str):
            return hmac.new(self.secret_key, message, 'sha256').digest()
        return hmac.new(self.secret_key, message.encode(), 'sha256').digest()

    def verify_hash(self, message, hash_value):
        """
        Verify the hash of the given message.
        """
        generated_hash = self.generate_hash(message)
        return hmac.compare_digest(generated_hash, hash_value)