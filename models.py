from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = "sqlite:///./fair_price.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Category(Base):
    __tablename__ = "categories"
    category_id = Column(Integer, primary_key=True, index=True)
    parent_category_id = Column(Integer, ForeignKey("categories.category_id"), nullable=True)
    category_name = Column(String, nullable=False, index=True)

class Product(Base):
    __tablename__ = "products"
    product_id = Column(Integer, primary_key=True, index=True)
    category_id = Column(Integer, ForeignKey("categories.category_id"), nullable=True)
    brand_name = Column(String, index=True)
    model_name = Column(String)
    msrp_price = Column(Float)
    specifications = Column(Text)

class Seller(Base):
    __tablename__ = "sellers"
    seller_id = Column(Integer, primary_key=True, index=True)
    seller_name = Column(String, nullable=False)
    website_url = Column(String)
    trust_score = Column(Float, default=5.0)

class ProductListing(Base):
    __tablename__ = "product_listings"
    listing_id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.product_id"))
    seller_id = Column(Integer, ForeignKey("sellers.seller_id"))
    current_price = Column(Float, nullable=False)
    discount_pct = Column(Float)
    listing_url = Column(String)
    last_scraped_at = Column(DateTime, default=datetime.utcnow)

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    role = Column(String, default="Shopper")
    phone_number = Column(String, nullable=True)
    date_of_birth = Column(String, nullable=True)

class PriceAlert(Base):
    __tablename__ = "price_alerts"
    alert_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    product_id = Column(Integer, ForeignKey("products.product_id"))
    target_price = Column(Float)
    status = Column(String, default="Active")

class PredictionQuery(Base):
    __tablename__ = "prediction_queries"
    query_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    listing_id = Column(Integer, ForeignKey("product_listings.listing_id"), nullable=True)
    model_id = Column(Integer, ForeignKey("model_artifacts.model_id"), nullable=True)
    evaluated_price = Column(Float, nullable=False)
    predicted_verdict = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    price_variance = Column(Float, nullable=False)
    query_timestamp = Column(DateTime, default=datetime.utcnow)

class ModelArtifact(Base):
    __tablename__ = "model_artifacts"
    model_id = Column(Integer, primary_key=True, index=True)
    version_number = Column(String, nullable=False)
    algorithm_type = Column(String, default="RandomForestRegressor")
    training_accuracy = Column(Float)
    is_active = Column(Boolean, default=True)

class Feedback(Base):
    __tablename__ = "feedbacks"
    feedback_id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("prediction_queries.query_id"))
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    is_accurate = Column(Boolean, nullable=False)
    user_comments = Column(Text, nullable=True)

class RetrainingLog(Base):
    __tablename__ = "retraining_logs"
    log_id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    model_id = Column(Integer, ForeignKey("model_artifacts.model_id"))
    data_sample_size = Column(Integer)
    execution_date = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)