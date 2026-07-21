import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

TIME_LABELS = ["day", "evening", "night"]
WEATHER_LABELS = ["clear", "cloudy", "partly_cloudy"]


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(history.history["time_accuracy"], label="Train")
    if "val_time_accuracy" in history.history:
        axes[0, 0].plot(history.history["val_time_accuracy"], label="Val")
    axes[0, 0].set_title("Time Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history.history["weather_accuracy"], label="Train")
    if "val_weather_accuracy" in history.history:
        axes[0, 1].plot(history.history["val_weather_accuracy"], label="Val")
    axes[0, 1].set_title("Weather Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history.history["time_loss"], label="Train")
    if "val_time_loss" in history.history:
        axes[1, 0].plot(history.history["val_time_loss"], label="Val")
    axes[1, 0].set_title("Time Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history.history["weather_loss"], label="Train")
    if "val_weather_loss" in history.history:
        axes[1, 1].plot(history.history["val_weather_loss"], label="Val")
    axes[1, 1].set_title("Weather Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def evaluate_model(model, test_gen):
    y_true_time = []
    y_pred_time = []
    y_true_weather = []
    y_pred_weather = []

    for i in range(len(test_gen)):
        result = test_gen[i]
        X, y = result[0], result[1]
        preds = model.predict(X, verbose=0)
        pred_time = np.argmax(preds[0], axis=1)
        pred_weather = np.argmax(preds[1], axis=1)
        true_time = np.argmax(y["time"], axis=1)
        true_weather = np.argmax(y["weather"], axis=1)

        # Only count labels that are known
        for j in range(len(true_time)):
            if np.any(y["time"][j] > 0):
                y_true_time.append(true_time[j])
                y_pred_time.append(pred_time[j])
            if np.any(y["weather"][j] > 0):
                y_true_weather.append(true_weather[j])
                y_pred_weather.append(pred_weather[j])

    results = {}

    if y_true_time:
        print("\n" + "=" * 50)
        print("Time Classification Report")
        print("=" * 50)
        results["time_report"] = classification_report(
            y_true_time, y_pred_time, target_names=TIME_LABELS, output_dict=True, zero_division=0
        )
        print(classification_report(y_true_time, y_pred_time, target_names=TIME_LABELS, zero_division=0))

        cm = confusion_matrix(y_true_time, y_pred_time)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=TIME_LABELS, yticklabels=TIME_LABELS, ax=ax)
        ax.set_title("Time Confusion Matrix")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        plt.show()

    if y_true_weather:
        print("\n" + "=" * 50)
        print("Weather Classification Report")
        print("=" * 50)
        results["weather_report"] = classification_report(
            y_true_weather, y_pred_weather, target_names=WEATHER_LABELS, output_dict=True, zero_division=0
        )
        print(classification_report(y_true_weather, y_pred_weather, target_names=WEATHER_LABELS, zero_division=0))

        cm = confusion_matrix(y_true_weather, y_pred_weather)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=WEATHER_LABELS, yticklabels=WEATHER_LABELS, ax=ax)
        ax.set_title("Weather Confusion Matrix")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        plt.show()

    return results
