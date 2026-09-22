# Data

`sample/` is generated deterministically by `visionguard generate-sample`. It contains synthetic
grayscale surfaces, three artificial defect types, and pixel masks.

The dataset exists to verify the anomaly-detection pipeline without redistributing third-party
data. It must not be used to claim industrial model quality.

The training split contains only normal images. The test split contains normal images and
synthetic scratches, spots, and cracks. `manifest.csv` records every image, mask, split, label,
and defect type.

The next milestone will benchmark on MVTec AD under its original license.

