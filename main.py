# main.py

from utils.train_utils import train_model

if __name__ == "__main__":
    train_model(
        images_dir='data/images',
        reports_dir='data/reports',
        save_dir='checkpoints',
        batch_size=16,
        learning_rate=1e-4,
        epochs=10
    )
