"""Machine-learning evidence for the technical analyst (LightGBM direction model).

    settings.py   config.yaml -> technical_model
    features.py   features + 3-class labels (down / flat / up)
    dataset.py    pooled point-in-time training table + price cache
    model.py      DirectionModel: LightGBM + temperature calibration + SHAP reasons
    validation.py purged walk-forward evaluation against baselines
    registry.py   picks a saved model valid for a date, or trains one
    predictor.py  MLDirectionService -> MLDirectionEvidence for the agent
    train.py      CLI: python -m ml.train
"""
