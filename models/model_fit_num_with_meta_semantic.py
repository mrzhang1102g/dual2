"""FIT numeric+meta model with auxiliary semantic supervision heads."""

from torch import nn

from models.model_fit_num_with_meta import Model_Fit_Num_With_Meta


class Model_Fit_Num_With_Meta_Semantic(Model_Fit_Num_With_Meta):
    """
    Keep the original FIT numeric+meta backbone unchanged and add lightweight
    semantic heads on top of the pooled summary state.
    """

    def __init__(self, configs):
        super().__init__(configs)

        hidden = getattr(configs, "semantic_hidden", self.d_model)
        dropout = getattr(configs, "semantic_dropout", self.dropout)

        self.semantic_trunk = nn.Sequential(
            nn.Linear(self.d_model, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.overall_head = nn.Linear(hidden, 3)
        self.recent_head = nn.Linear(hidden, 3)
        self.volatility_head = nn.Linear(hidden, 3)
        self.turning_head = nn.Linear(hidden, 4)

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        city_ids,
        gender_ids,
        age_ids,
        element_ids,
    ):
        features = self.extract_features(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            city_ids=city_ids,
            gender_ids=gender_ids,
            age_ids=age_ids,
            element_ids=element_ids,
        )

        semantic_state = self.semantic_trunk(features["summary_state"])
        return {
            "forecast": features["forecast"],
            "semantic_logits": {
                "overall": self.overall_head(semantic_state),
                "recent": self.recent_head(semantic_state),
                "volatility": self.volatility_head(semantic_state),
                "turning": self.turning_head(semantic_state),
            },
        }
