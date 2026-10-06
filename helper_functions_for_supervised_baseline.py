def create_supervised_baseline_from_checkpoint_config(
    checkpoint_path: Union[str, Path],
    num_classes: int,
    *,
    device: Union[str, torch.device] = "auto",
) -> nn.Module:
    """
    Create a randomly initialized ECG_VCG_ResNet with the same architecture
    as the pretrained checkpoint, but WITHOUT loading pretrained weights.

    This is the supervised-from-scratch baseline.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    if device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    if not isinstance(cfg, dict):
        raise TypeError("Checkpoint 'config' must be a dictionary.")

    task_type = str(cfg.get("task_type", "classification"))
    use_vcg = bool(cfg.get("use_vcg", True))
    use_tabular = bool(cfg.get("use_tabular", True))

    if task_type != "classification":
        raise ValueError(
            f"Expected classification checkpoint configuration, got {task_type!r}."
        )

    model_kwargs = {
        "feature_dim": int(cfg.get("feature_dim", 256)),
        "num_classes": int(num_classes),
        "task_type": "classification",
        "filter_size": int(cfg.get("filter_size", 16)),
        "input_channels": int(cfg.get("input_channels", 12)),
        "dropout_value": float(cfg.get("dropout_value", 0.5)),
        "conv1_kernel_size": int(cfg.get("conv1_kernel_size", 15)),
        "conv1_stride": int(cfg.get("conv1_stride", 2)),
        "conv1_padding": int(cfg.get("conv1_padding", 7)),
        "use_vcg": use_vcg,
        "vcg_method": str(cfg.get("vcg_method", "kors")),
        "vcg_projection": cfg.get("vcg_projection", None),
        "vcg_normalization": str(cfg.get("vcg_normalization", "rms")),
        "use_vcg_velocity": bool(cfg.get("use_vcg_velocity", True)),
        "vcg_velocity_method": str(cfg.get("vcg_velocity_method", "central")),
        "vcg_velocity_dt": float(cfg.get("vcg_velocity_dt", 1.0)),
        "vcg_velocity_smooth_window": int(
            cfg.get("vcg_velocity_smooth_window", 0)
        ),
        "gate_mode": str(cfg.get("gate_mode", "scalar")),
        "use_learned_gate": bool(cfg.get("use_learned_gate", True)),
        "use_tabular": use_tabular,
        "tabular_dim": int(cfg.get("tabular_dim", 7)),
        "use_aux_loss": bool(cfg.get("use_aux_loss", True)),
        "lambda_vcg": float(cfg.get("lambda_vcg", 0.1)),
        "contrastive_temperature": float(
            cfg.get("contrastive_temperature", 0.08)
        ),
    }

    model = ECG_VCG_ResNet(**model_kwargs).to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(True)

    n_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    n_total = sum(p.numel() for p in model.parameters())

    print("Created supervised baseline with RANDOM initialization.")
    print(f"Number of classes: {num_classes}")
    print(f"Use VCG: {use_vcg}")
    print(f"Use tabular: {use_tabular}")
    print(f"Trainable parameters: {n_trainable:,}/{n_total:,}")

    return model


def train_supervised_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: Union[str, torch.device] = "cuda",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
):
    """
    Train the complete model from random initialization.
    """
    if device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    model = model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = np.inf
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()

        train_loss_sum = 0.0
        train_n = 0

        for x, t, y in train_loader:
            x = x.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(x,x_tabular=t)

            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            batch_size = y.shape[0]
            train_loss_sum += float(loss.item()) * batch_size
            train_n += batch_size

        train_loss = train_loss_sum / train_n

        model.eval()

        val_loss_sum = 0.0
        val_n = 0

        with torch.inference_mode():
            for x, t, y in val_loader:
                x = x.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x,x_tabular=t,)
                loss = criterion(logits, y)

                batch_size = y.shape[0]
                val_loss_sum += float(loss.item()) * batch_size
                val_n += batch_size

        val_loss = val_loss_sum / val_n

        print(f"Epoch {epoch + 1:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_state = {key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model = model.to(device)
    model.eval()

    return model


model_supervised = create_supervised_baseline_from_checkpoint_config(
    CHECKPOINT_PATH,
    num_classes=len(finetune_label_cols),
    device="cuda",
)

loaders = create_finetune_dataloaders(
    train_waveforms_path=RV_TRAIN_WAVEFORMS,
    val_waveforms_path=RV_VAL_WAVEFORMS,
    test_waveforms_path=RV_TEST_WAVEFORMS,
    train_tabular_path=RV_TRAIN_TABULAR,
    val_tabular_path=RV_VAL_TABULAR,
    test_tabular_path=RV_TEST_TABULAR,
    metadata_csv=RV_METADATA_CSV,
    label_cols=finetune_label_cols,
    train_indices=nested_indices[0.05],
    batch_size=64,
    num_workers=2,
)

model_supervised = train_supervised_baseline(
    model_supervised,
    loaders["train"],
    loaders["val"],
    device="cuda",
    epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    patience=7,
)

supervised_test_metrics_5pct = evaluate_multilabel_classifier(
    model_supervised,
    loaders["test"],
    device="auto",
    label_cols=finetune_label_cols,
    use_tabular=True,
)

print(f"Test macro AUROC: "
    f"{supervised_test_metrics_5pct['macro_auroc']:.4f}")
print(f"Test macro AUPRC: "
    f"{supervised_test_metrics_5pct['macro_auprc']:.4f}")

display(supervised_test_metrics_5pct["per_class"])


fig_auroc, fig_auprc = plot_transfer_learning_curves(
    linear_probe_results=results_linear_probe,
    finetune_results=results_full_finetune,
    supervised_results=results_supervised,
    label_fractions=label_fractions,
    plot_mode="per_class",
)
