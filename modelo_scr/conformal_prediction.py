import pickle
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# BASE SCORE FUNCTION
# ============================================================

class BaseScoreFunction:

    requires_uncertainty = False

    def __call__(
        self,
        y_true,
        y_pred,
        std=None,
        eps=1e-8
    ):
        raise NotImplementedError

    def prediction_interval(
        self,
        y_pred,
        qhat,
        std=None,
        eps=1e-8
    ):
        raise NotImplementedError


# ============================================================
# ABSOLUTE SCORE
# ============================================================

class AbsoluteScore(BaseScoreFunction):

    requires_uncertainty = False

    def __call__(
        self,
        y_true,
        y_pred,
        std=None,
        eps=1e-8
    ):

        return np.abs(
            y_true - y_pred
        )

    def prediction_interval(
        self,
        y_pred,
        qhat,
        std=None,
        eps=1e-8
    ):

        lower = y_pred - qhat
        upper = y_pred + qhat

        return lower, upper


# ============================================================
# NORMALIZED SCORE
# ============================================================

class NormalizedScore(BaseScoreFunction):

    requires_uncertainty = True

    def __call__(
        self,
        y_true,
        y_pred,
        std=None,
        eps=1e-8
    ):

        return (
            np.abs(y_true - y_pred)
            / (std + eps)
        )

    def prediction_interval(
        self,
        y_pred,
        qhat,
        std=None,
        eps=1e-8
    ):

        lower = (
            y_pred
            - qhat * (std + eps)
        )

        upper = (
            y_pred
            + qhat * (std + eps)
        )

        return lower, upper


# ============================================================
# VARIANCE SCALED SCORE
# ============================================================

class VarianceScaledScore(BaseScoreFunction):

    requires_uncertainty = True

    def __call__(
        self,
        y_true,
        y_pred,
        std=None,
        eps=1e-8
    ):

        variance = (std ** 2) + eps

        return (
            ((y_true - y_pred) ** 2)
            / variance
        )

    def prediction_interval(
        self,
        y_pred,
        qhat,
        std=None,
        eps=1e-8
    ):

        scale = np.sqrt(qhat) * (std + eps)

        lower = y_pred - scale
        upper = y_pred + scale

        return lower, upper


# ============================================================
# SCORE REGISTRY
# ============================================================

SCORE_FUNCTIONS = {
    "absolute": AbsoluteScore(),
    "normalized": NormalizedScore(),
    "variance_scaled": VarianceScaledScore()
}


# ============================================================
# CONFORMAL REGRESSOR
# ============================================================

class ConformalRegressor:

    def __init__(
        self,
        model,
        X_calib,
        y_calib,
        score_function="absolute",
        alpha=0.1,
        eps=1e-8,
        verbose=True
    ):

        # ----------------------------------------------------
        # MODEL MUST IMPLEMENT:
        #
        # predict(X)
        #
        # RETURN:
        #
        # 1)
        # y_pred
        #
        # OR
        #
        # 2)
        # y_pred, std
        # ----------------------------------------------------

        self.model = model
        self.alpha = alpha
        self.eps = eps
        self.qhat = None
        self.score_function = None
        self.score_name = None
        self.calibration_scores = None
        self.is_calibrated = False

        # ----------------------------------------------------
        # CALIBRATION DATA
        # ----------------------------------------------------

        self.X_calib = None

        self.y_calib = None

        if X_calib is not None and y_calib is not None:

            self.calibrate(
                X_calib=X_calib,
                y_calib=y_calib,
                score_function=score_function
            )
        else:
            print(
                "ConformalRegressor initialized "
                "without calibration data."
            )

    # ========================================================
    # INTERNAL PREDICTION PARSER
    # ========================================================

    def _parse_prediction_output(
        self,
        prediction_output
    ):

        # ----------------------------------------------------
        # CASE 1
        # predict() returns only y_pred
        # ----------------------------------------------------

        if not isinstance(
            prediction_output,
            tuple
        ):

            y_pred = np.asarray(
                prediction_output
            ).flatten()

            std = None

            return y_pred, std

        # ----------------------------------------------------
        # CASE 2
        # predict() returns:
        # (y_pred, std)
        # ----------------------------------------------------

        if len(prediction_output) != 2:

            raise ValueError(
                "predict() must return:\n"
                "- y_pred\n"
                "or\n"
                "- (y_pred, std)"
            )

        y_pred, std = prediction_output

        y_pred = np.asarray(
            y_pred
        ).flatten()

        std = np.asarray(
            std
        ).flatten()

        # ----------------------------------------------------
        # VALIDATE STD
        # ----------------------------------------------------

        if np.any(std < 0):

            raise ValueError(
                "Standard deviation "
                "(std) must be non-negative."
            )

        return y_pred, std

    # ========================================================
    # SCORE FUNCTION LOADER
    # ========================================================

    def _get_score_function(
        self,
        score_function
    ):

        # ----------------------------------------------------
        # BUILT-IN SCORE
        # ----------------------------------------------------

        if isinstance(
            score_function,
            str
        ):

            if score_function not in SCORE_FUNCTIONS:

                raise ValueError(
                    f"Unknown score function: "
                    f"{score_function}"
                )

            return SCORE_FUNCTIONS[
                score_function
            ]

        # ----------------------------------------------------
        # CUSTOM SCORE OBJECT
        # ----------------------------------------------------

        if not hasattr(
            score_function,
            "__call__"
        ):

            raise ValueError(
                "Custom score function "
                "must implement __call__."
            )

        return score_function

    # ========================================================
    # CALIBRATION
    # ========================================================

    def calibrate(
        self,
        X_calib,
        y_calib,
        score_function="absolute"
    ):

        # ----------------------------------------------------
        # STORE CALIBRATION DATA
        # ----------------------------------------------------

        self.X_calib = np.asarray(
            X_calib
        )

        self.y_calib = np.asarray(
            y_calib
        )

        # ----------------------------------------------------
        # LOAD SCORE FUNCTION
        # ----------------------------------------------------

        self.score_function = (
            self._get_score_function(
                score_function
            )
        )

        self.score_name = (
            score_function
            if isinstance(score_function, str)
            else score_function.__class__.__name__
        )

        # ----------------------------------------------------
        # MODEL PREDICTIONS
        # ----------------------------------------------------

        prediction_output = (
            self.model.predict(X_calib)
        )

        y_pred, std = (
            self._parse_prediction_output(
                prediction_output
            )
        )

        # ----------------------------------------------------
        # VALIDATE UNCERTAINTY
        # ----------------------------------------------------

        if (
            self.score_function.requires_uncertainty
            and std is None
        ):

            raise ValueError(
                f"Score function "
                f"'{self.score_name}' "
                f"requires predict() "
                f"to return:\n"
                f"(y_pred, std)"
            )

        # ----------------------------------------------------
        # CALCULATE SCORES
        # ----------------------------------------------------

        scores = self.score_function(
            y_true=y_calib,
            y_pred=y_pred,
            std=std,
            eps=self.eps
        )

        self.calibration_scores = scores

        # ----------------------------------------------------
        # CONFORMAL QUANTILE
        # ----------------------------------------------------

        n = len(scores)

        q_level = (
            np.ceil(
                (n + 1)
                * (1 - self.alpha)
            )
            / n
        )

        q_level = min(q_level, 1.0)

        self.qhat = np.quantile(
            scores,
            q_level,
            method="higher"
        )

        self.is_calibrated = True

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        X
    ):

        if not self.is_calibrated:

            raise ValueError(
                "Conformal model "
                "is not calibrated."
            )

        prediction_output = (
            self.model.predict(X)
        )

        y_pred, std = (
            self._parse_prediction_output(
                prediction_output
            )
        )

        # ----------------------------------------------------
        # VALIDATE UNCERTAINTY
        # ----------------------------------------------------

        if (
            self.score_function.requires_uncertainty
            and std is None
        ):

            raise ValueError(
                f"Score function "
                f"'{self.score_name}' "
                f"requires predict() "
                f"to return:\n"
                f"(y_pred, std)"
            )

        # ----------------------------------------------------
        # PREDICTION INTERVAL
        # ----------------------------------------------------

        lower, upper = (
            self.score_function
            .prediction_interval(
                y_pred=y_pred,
                qhat=self.qhat,
                std=std,
                eps=self.eps
            )
        )

        return (
            y_pred,
            lower,
            upper
        )

    # ========================================================
    # EVALUATE
    # ========================================================

    def evaluate(
        self,
        X_test,
        y_test
    ):

        y_pred, lower, upper = (
            self.predict(X_test)
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        coverage = np.mean(
            (y_test >= lower)
            &
            (y_test <= upper)
        )

        mpiw = np.mean(
            upper - lower
        )

        coverage_gap = abs(
            coverage
            - (1 - self.alpha)
        )

        return {
            "MAE": mae,
            "RMSE": rmse,
            "Coverage": coverage,
            "MPIW": mpiw,
            "CoverageGap": coverage_gap,
            "Qhat": self.qhat,
            "Alpha": self.alpha,
            "ScoreFunction": self.score_name
        }

    # ========================================================
    # SAVE CALIBRATION
    # ========================================================

    def save_calibration(
        self,
        filepath
    ):

        if not self.is_calibrated:

            raise ValueError(
                "Cannot save calibration "
                "before calibrating."
            )

        calibration_data = {

            # ------------------------------------------------
            # CONFIGURATION
            # ------------------------------------------------

            "alpha": self.alpha,

            "eps": self.eps,

            "score_name": self.score_name,

            # ------------------------------------------------
            # CALIBRATION RESULTS
            # ------------------------------------------------

            "qhat": self.qhat,

            "calibration_scores":
                self.calibration_scores,

            # ------------------------------------------------
            # CALIBRATION DATASET
            # ------------------------------------------------

            "X_calib": self.X_calib,

            "y_calib": self.y_calib,

            # ------------------------------------------------
            # STATE
            # ------------------------------------------------

            "is_calibrated":
                self.is_calibrated
        }

        with open(filepath, "wb") as f:

            pickle.dump(
                calibration_data,
                f
            )

    # ========================================================
    # LOAD CALIBRATION
    # ========================================================

    def load_calibration(
        self,
        filepath
    ):

        with open(filepath, "rb") as f:

            calibration_data = pickle.load(f)

        # ----------------------------------------------------
        # CONFIGURATION
        # ----------------------------------------------------

        self.alpha = calibration_data["alpha"]

        self.eps = calibration_data["eps"]

        self.score_name = (
            calibration_data["score_name"]
        )

        # ----------------------------------------------------
        # CALIBRATION RESULTS
        # ----------------------------------------------------

        self.qhat = calibration_data["qhat"]

        self.calibration_scores = (
            calibration_data[
                "calibration_scores"
            ]
        )

        # ----------------------------------------------------
        # CALIBRATION DATASET
        # ----------------------------------------------------

        self.X_calib = (
            calibration_data["X_calib"]
        )

        self.y_calib = (
            calibration_data["y_calib"]
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self.is_calibrated = (
            calibration_data[
                "is_calibrated"
            ]
        )

        # ----------------------------------------------------
        # RESTORE SCORE FUNCTION
        # ----------------------------------------------------

        if (
            self.score_name
            not in SCORE_FUNCTIONS
        ):

            raise ValueError(
                f"Unknown score function "
                f"stored in calibration: "
                f"{self.score_name}"
            )

        self.score_function = (
            SCORE_FUNCTIONS[
                self.score_name
            ]
        )


class CPCalibration:
    def __init__(self, model, dataset_calib, mc=True, alpha=0.05):
        self.model = model
        self.dataset_calib = dataset_calib
        self.mc = mc
        self.alpha = alpha

        if self.mc:
            self.cp_dict =  {"absolute": ConformalRegressor(self.model,
                                    X_calib=dataset_calib.X_measured_mc.reshape(-1, 1).detach().cpu().numpy(),
                                    y_calib=dataset_calib.y_measured_mc.detach().cpu().numpy(),
                                    score_function="absolute",
                                    alpha=alpha),
                            "normalized": ConformalRegressor(self.model,
                                    X_calib=dataset_calib.X_measured_mc.reshape(-1, 1).detach().cpu().numpy(),
                                    y_calib=dataset_calib.y_measured_mc.detach().cpu().numpy(),
                                    score_function="normalized",
                                    alpha=alpha)}
        else:
            self.cp_dict =  {"absolute": ConformalRegressor(self.model,
                                    X_calib=dataset_calib.X_measured.reshape(-1, 1).detach().cpu().numpy(),
                                    y_calib=dataset_calib.y_measured.detach().cpu().numpy(),
                                    score_function="absolute",
                                    alpha=alpha),
                            "normalized": ConformalRegressor(self.model,
                                    X_calib=dataset_calib.X_measured.reshape(-1, 1).detach().cpu().numpy(),
                                    y_calib=dataset_calib.y_measured.detach().cpu().numpy(),
                                    score_function="normalized",
                                    alpha=alpha)}