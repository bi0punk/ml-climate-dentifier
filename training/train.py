import os
import argparse
from datetime import datetime
import json

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from training.model import get_model, compile_model, IMG_SIZE
from training.dataset import get_train_val_generators, create_splits
from training.evaluate import plot_training_history, evaluate_model


def train(args):
    print("=" * 60)
    print("ML Climate Dentifier - Training Pipeline v2")
    print("=" * 60)

    train_gen, val_gen = get_train_val_generators(args.data_dir, batch_size=args.batch_size)

    print(f"Train batches: {len(train_gen)}, Val batches: {len(val_gen)}")
    print(f"Model backbone: {args.backbone}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.learning_rate}")

    model = get_model(backbone=args.backbone, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model = compile_model(model, learning_rate=args.learning_rate)
    model.summary()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, f"best_model_{timestamp}.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=args.patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
        CSVLogger(
            filename=os.path.join(model_dir, f"training_log_{timestamp}.csv"),
        ),
    ]

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    final_path = os.path.join(model_dir, f"weather_classifier_{timestamp}.h5")
    model.save(final_path)
    print(f"\nModel saved: {final_path}")

    plot_training_history(history, save_path=os.path.join(model_dir, f"training_plot_{timestamp}.png"))

    if args.evaluate:
        _, _, test_gen = create_splits(args.data_dir, batch_size=args.batch_size)
        evaluate_model(model, test_gen)

    config = vars(args)
    config["final_model_path"] = final_path
    with open(os.path.join(model_dir, f"config_{timestamp}.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Train weather/time classifier v2")
    parser.add_argument("--data-dir", default="dataset", help="Path to raw dataset")
    parser.add_argument("--model-dir", default="models", help="Where to save models")
    parser.add_argument("--backbone", default="cnn", choices=["cnn", "mobilenetv2"], help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on test set after training")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
