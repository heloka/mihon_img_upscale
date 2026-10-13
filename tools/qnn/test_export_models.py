import unittest
from pathlib import Path

import torch

from export_models import DeconvolutionAsConv, RealEsrgan, realcugan_display_range


class RealCuganDisplayRangeTest(unittest.TestCase):
    def test_pro_inverse_normalization_matches_upstream_range(self):
        network_output = torch.tensor([0.0, 0.15, 0.5, 0.85, 1.0])

        actual = realcugan_display_range(network_output, is_pro=True)

        expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
        torch.testing.assert_close(actual, expected)

    def test_se_output_stays_in_its_native_display_range(self):
        network_output = torch.tensor([0.0, 0.15, 0.5, 0.85, 1.0])

        actual = realcugan_display_range(network_output, is_pro=False)

        torch.testing.assert_close(actual, network_output)


class DeconvolutionRewriteTest(unittest.TestCase):
    def test_conv_rewrite_matches_kernel_two_transposed_convolution(self):
        torch.manual_seed(17)
        deconvolution = torch.nn.ConvTranspose2d(5, 3, kernel_size=2, stride=2)
        rewritten = DeconvolutionAsConv(deconvolution)
        image = torch.randn(2, 5, 11, 13)

        torch.testing.assert_close(rewritten(image), deconvolution(image), rtol=1e-6, atol=1e-6)

    def test_conv_rewrite_matches_padded_stride_two_transposed_convolution(self):
        torch.manual_seed(23)
        deconvolution = torch.nn.ConvTranspose2d(4, 3, kernel_size=4, stride=2, padding=3)
        rewritten = DeconvolutionAsConv(deconvolution)
        image = torch.randn(1, 4, 9, 11)

        torch.testing.assert_close(rewritten(image), deconvolution(image), rtol=1e-6, atol=1e-6)


class RealEsrganNcnnImportTest(unittest.TestCase):
    def test_anime_x2_ncnn_weights_load_and_export_finite_x2_output(self):
        model_prefix = (
            Path(__file__).parents[2]
            / "app"
            / "src"
            / "main"
            / "assets"
            / "realesrgan-models"
            / "v3-anime"
            / "x2"
        )

        model = RealEsrgan(scale=2, num_conv=16, ncnn_model=model_prefix).eval()
        image = torch.linspace(0, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
        with torch.inference_mode():
            result = model(image)

        self.assertEqual(tuple(result.shape), (1, 3, 64, 64))
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(torch.allclose(model.model.body[1].weight, torch.full_like(model.model.body[1].weight, 0.25)))

if __name__ == "__main__":
    unittest.main()
