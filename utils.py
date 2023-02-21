class Tiers:
    free: str = 'Free'
    basic: str = 'Basic'
    premium: str = 'Premium'

    @classmethod
    def get_rate_limit(cls, tier: str) -> str:
        if tier == cls.free:
            return "1/minute; 100/hour"
        elif tier == cls.basic:
            return "100/minute"
        elif tier == cls.premium:
            return "1000/minute"
        else:
            raise Exception(f"Unknown tier: {tier}")


class User:
    def __init__(self, user_id: int = None, username: str = None, email: str = None, tier: str = Tiers.free, **data):
        super().__init__(**data)
        self.user_id = user_id
        self.username = username
        self.email = email
        self.tier = tier

    @classmethod
    def anonymous(cls):
        return User()

    @classmethod
    def from_jwt(cls, decoded_jwt: dict):
        if not decoded_jwt:
            return cls.anonymous()
        user_id = decoded_jwt.get('user_id')
        if not user_id:
            return cls.anonymous()
        username = decoded_jwt.get('username')
        email = decoded_jwt.get('email')
        tier = decoded_jwt.get('tier')
        user = User(user_id=user_id, username=username, email=email, tier=tier)
        return user

    def is_authenticated(self) -> bool:
        return self.user_id is not None

    def get_rate_limit(self) -> str:
        return Tiers.get_rate_limit(self.tier)
