import os
from typing import Optional, Union, Tuple
import torch
from PIL import Image
import numpy as np

from .config import AgeProgressionConfig
from .models.full_model import AgeProgressionModel
from .data.transforms import RetinaFacePreprocess


class AgeProgressionPipeline:
    """
    Inference Pipeline for Age Progression with Kinship Guidance.
    Takes child face photo, optional parent photos, and target age,
    and synthesizes the predicted face image.
    """

    def __init__(
        self,
        model: Optional[AgeProgressionModel] = None,
        config: Optional[AgeProgressionConfig] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.config = config or AgeProgressionConfig(device=device)

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = AgeProgressionModel(self.config).to(self.device)
            if checkpoint_path and os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)

        self.model.eval()
        self.preprocessor = RetinaFacePreprocess(target_size=self.config.image_size)

    def _prepare_tensor(self, img_input: Optional[Union[str, Image.Image, torch.Tensor]]) -> Optional[torch.Tensor]:
        if img_input is None:
            return None
        if isinstance(img_input, str):
            img_input = Image.open(img_input).convert("RGB")
        if isinstance(img_input, Image.Image):
            tensor = self.preprocessor.align_and_crop(img_input)
            return tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)
        if isinstance(img_input, torch.Tensor):
            if img_input.dim() == 3:
                img_input = img_input.unsqueeze(0)
            return img_input.to(self.device)
        return None

    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Denormalizes tensor in [-1, 1] to uint8 PIL Image."""
        t = tensor.detach().cpu().squeeze(0)  # (3, H, W)
        t = (t * 0.5 + 0.5).clamp(0, 1)        # [0, 1]
        arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    @torch.no_grad()
    def predict(
        self,
        child: Union[str, Image.Image, torch.Tensor],
        target_age: float,
        father: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        mother: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        source_age: Optional[float] = None
    ) -> Tuple[Image.Image, dict]:
        """
        Executes age progression.
        
        Args:
            child: Child face (path, PIL Image, or tensor)
            target_age: Goal chronological age (e.g. 25)
            father: Optional father face
            mother: Optional mother face
            source_age: Optional child source age (e.g. 5)
        Returns:
            aged_image: PIL Image of predicted face at target age
            metadata: dictionary of intermediate embeddings and attention weights
        """
        child_tensor = self._prepare_tensor(child)
        father_tensor = self._prepare_tensor(father)
        mother_tensor = self._prepare_tensor(mother)
        
        target_age_tensor = torch.tensor([float(target_age)], device=self.device)
        source_age_tensor = torch.tensor([float(source_age)], device=self.device) if source_age else None

        outputs = self.model(
            child_img=child_tensor,
            target_age=target_age_tensor,
            father_img=father_tensor,
            mother_img=mother_tensor,
            source_age=source_age_tensor
        )

        aged_image = self._tensor_to_image(outputs["pred_img"])
        metadata = {
            "target_age": target_age,
            "attn_weights": outputs.get("attn_weights"),
            "v_kin_norm": float(torch.norm(outputs["v_kin"]).item())
        }
        return aged_image, metadata

