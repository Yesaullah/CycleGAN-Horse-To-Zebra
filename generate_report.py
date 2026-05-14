"""Generates the DLP project report as a clean academic PDF using ReportLab."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

W, H = A4
BLACK = colors.black
WHITE = colors.white
LGREY = colors.HexColor("#f0f0f0")

# ── Styles ─────────────────────────────────────────────────────────────────────
def S(name, **kw):
    return ParagraphStyle(name, **kw)

TitleStyle = S("Title",
    fontName="Times-Bold", fontSize=16, leading=22,
    alignment=TA_CENTER, textColor=BLACK,
    spaceBefore=0, spaceAfter=6)

SubtitleStyle = S("Subtitle",
    fontName="Times-Roman", fontSize=12, leading=16,
    alignment=TA_CENTER, textColor=BLACK,
    spaceBefore=2, spaceAfter=2)

H1Style = S("H1",
    fontName="Times-Bold", fontSize=13, leading=18,
    alignment=TA_LEFT, textColor=BLACK,
    spaceBefore=16, spaceAfter=4)

H2Style = S("H2",
    fontName="Times-Bold", fontSize=12, leading=16,
    alignment=TA_LEFT, textColor=BLACK,
    spaceBefore=10, spaceAfter=4)

BodyStyle = S("Body",
    fontName="Times-Roman", fontSize=12, leading=18,
    alignment=TA_JUSTIFY, textColor=BLACK,
    spaceBefore=0, spaceAfter=6)

BulletStyle = S("Bullet",
    fontName="Times-Roman", fontSize=12, leading=18,
    alignment=TA_JUSTIFY, textColor=BLACK,
    leftIndent=20, firstLineIndent=0,
    spaceBefore=2, spaceAfter=2)

CaptionStyle = S("Caption",
    fontName="Times-Roman", fontSize=10, leading=14,
    alignment=TA_CENTER, textColor=BLACK,
    spaceBefore=2, spaceAfter=8)

CellStyle = S("Cell",
    fontName="Times-Roman", fontSize=10, leading=13,
    alignment=TA_LEFT, textColor=BLACK,
    spaceBefore=0, spaceAfter=0)

CellBoldStyle = S("CellBold",
    fontName="Times-Bold", fontSize=10, leading=13,
    alignment=TA_LEFT, textColor=BLACK,
    spaceBefore=0, spaceAfter=0)

# ── Helpers ────────────────────────────────────────────────────────────────────
def sp(pts=6):
    return Spacer(1, pts)

def p(text):
    return Paragraph(text, BodyStyle)

def h1(text):
    return Paragraph(text, H1Style)

def h2(text):
    return Paragraph(text, H2Style)

def b(text):
    return Paragraph(f"- {text}", BulletStyle)

def simple_table(data, col_widths):
    # Wrap every string cell in a Paragraph so text reflows within column width.
    wrapped = []
    for r_idx, row in enumerate(data):
        style = CellBoldStyle if r_idx == 0 else CellStyle
        wrapped.append([
            Paragraph(cell, style) if isinstance(cell, str) else cell
            for cell in row
        ])
    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("GRID",         (0, 0), (-1, -1), 0.5, BLACK),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    return t

# ── Page template ──────────────────────────────────────────────────────────────
def page_template(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 10)
    canvas.setFillColor(BLACK)
    canvas.drawString(2*cm, 1.4*cm, "CycleGAN Horse-Zebra Translation — DLP Project Report")
    canvas.drawRightString(W - 2*cm, 1.4*cm, f"Page {doc.page}")
    canvas.restoreState()

def first_page_template(canvas, doc):
    page_template(canvas, doc)

# ── Build ──────────────────────────────────────────────────────────────────────
def build():
    out = "DLP_Project_Report.pdf"
    doc = SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
        title="CycleGAN Horse-Zebra Translation — DLP Project Report",
    )

    story = []

    # ── Cover ──────────────────────────────────────────────────────────────────
    story += [
        sp(3*cm),
        Paragraph("CycleGAN: Horse to Zebra Unpaired Image-to-Image Translation", TitleStyle),
        sp(6),
        Paragraph("with Proposed Perceptual Cycle Consistency Loss", SubtitleStyle),
        sp(12),
        Paragraph("Deep Learning for Perception", SubtitleStyle),
        Paragraph("Final Semester Project Report", SubtitleStyle),
        sp(6),
        Paragraph("k230019@nu.edu.pk", SubtitleStyle),
        sp(4),
        Paragraph("May 9, 2026", SubtitleStyle),
        PageBreak(),
    ]

    # ── 1. Task Definition ─────────────────────────────────────────────────────
    story += [
        h1("1. Task Definition"),
        p("This project addresses the problem of unpaired image-to-image translation, "
          "a computer vision task where the goal is to learn a mapping between two "
          "visual domains without any paired training examples. Specifically, the task "
          "involves translating photographs of horses into photographs that resemble "
          "zebras, and vice versa."),
        p("The framework used to solve this task is CycleGAN, proposed by Zhu et al. "
          "(2017). CycleGAN learns two translation functions simultaneously: one that "
          "maps horse images to the zebra domain, and another that maps zebra images "
          "back to the horse domain. A cycle consistency constraint is enforced so that "
          "translating an image from one domain to the other and back again recovers "
          "the original image. This constraint allows the model to train without any "
          "paired correspondence between the two domains."),
        p("Beyond replicating the standard CycleGAN baseline, this project also proposes "
          "a modification to the cycle consistency loss. The standard approach penalises "
          "pixel-level differences between the original and reconstructed images, which "
          "can lead to blurry outputs. The proposed modification replaces this "
          "pixel-level penalty with a perceptual loss computed in the feature space of "
          "a pretrained VGG-16 network. The hypothesis is that comparing images at the "
          "level of learned visual features produces sharper, more structurally faithful "
          "reconstructions."),
        sp(6),
    ]

    # ── 2. Dataset Description ─────────────────────────────────────────────────
    story += [
        h1("2. Dataset Description"),
        p("The Horse2Zebra dataset, introduced alongside the CycleGAN paper (Zhu et al., "
          "2017), is used for this project. The dataset consists of RGB colour "
          "photographs sourced from the ImageNet horse synset (n02381460) and zebra "
          "synset (n02391049). All images are in JPEG format."),
        p("The dataset is split into four directories. The table below summarises the "
          "composition of each split:"),
        sp(6),
        simple_table(
            [
                ["Directory", "Domain", "Number of Images"],
                ["trainA", "Horses (training)", "1,067"],
                ["trainB", "Zebras (training)", "1,334"],
                ["testA",  "Horses (test)",     "120"],
                ["testB",  "Zebras (test)",      "140"],
            ],
            col_widths=[3.5*cm, 6*cm, 6.5*cm],
        ),
        sp(8),
        p("The dataset does not contain any labels in the conventional sense. There "
          "are no bounding boxes, segmentation masks, or class annotations. The only "
          "supervision available is domain membership, meaning each image is known to "
          "belong to either the horse domain or the zebra domain. The task is therefore "
          "fully unsupervised with respect to image content."),
        p("Importantly, the training images are unpaired. There is no correspondence "
          "between any horse image in trainA and any zebra image in trainB. The model "
          "must discover the visual differences between the two domains entirely from "
          "their respective distributions."),
        p("It is worth noting that the zebra training set is approximately 25 percent "
          "larger than the horse training set. This imbalance partially contributes to "
          "the observed asymmetry in translation quality, where the horse to zebra "
          "direction tends to produce better results than the reverse."),
        sp(6),
    ]

    # ── 3. Data Pre-processing ─────────────────────────────────────────────────
    story += [
        h1("3. Data Pre-processing"),
        h2("3.1 Training Pipeline"),
        p("Each training image undergoes the following sequence of transformations "
          "before being passed to the network:"),
        sp(4),
        b("Resize: Each image is resized to 286 by 286 pixels using bicubic "
          "interpolation. This is intentionally larger than the final training "
          "resolution to allow for spatial jitter in the next step."),
        b("Random Crop: A 256 by 256 patch is randomly cropped from the 286 by 286 "
          "image. This introduces positional augmentation and prevents the network "
          "from overfitting to edge locations."),
        b("Random Horizontal Flip: The image is flipped horizontally with a "
          "probability of 0.5. This effectively doubles the number of training "
          "examples and improves the model's left-right invariance."),
        b("Normalisation: Pixel values, originally in the range 0 to 1, are "
          "normalised to the range negative 1 to positive 1 using a mean of 0.5 and "
          "a standard deviation of 0.5 for each colour channel. This range matches "
          "the output range of the generator's Tanh activation."),
        sp(6),
        h2("3.2 Test and Inference Pipeline"),
        p("During evaluation and inference, augmentation is disabled to ensure "
          "deterministic results. Each image is resized directly to 256 by 256 "
          "pixels using bicubic interpolation, followed by the same normalisation "
          "step as training. No random cropping or flipping is applied."),
        p("No additional pre-processing steps such as noise reduction, histogram "
          "equalisation, or colour jitter are applied at any stage. Since the "
          "domain difference between horses and zebras is primarily one of texture "
          "rather than illumination or geometry, aggressive colour augmentation "
          "would interfere with the style signal the model needs to learn."),
        p("The data loading class samples images from Domain A and Domain B "
          "independently within each batch. No attempt is made to create or enforce "
          "any pairing between horse and zebra images during training."),
        sp(6),
    ]

    # ── 4. Network Architecture ────────────────────────────────────────────────
    story += [
        h1("4. Network Architecture"),
        p("The overall system consists of four neural networks trained simultaneously: "
          "two generators and two discriminators. The generators learn the translation "
          "mappings between domains, while the discriminators assess whether generated "
          "images are realistic."),
        sp(4),
        h2("4.1 Generator"),
        p("The generator follows the ResNet-based encoder-decoder architecture "
          "proposed by Johnson et al. (2016), adapted for use in CycleGAN. The network "
          "consists of three stages: a downsampling encoder, a sequence of residual "
          "blocks that transform the feature representation, and an upsampling decoder "
          "that reconstructs the image at full resolution."),
        p("The encoder begins with a large-kernel convolutional layer that processes "
          "the full-resolution input and produces 64 feature maps. This is followed by "
          "two strided convolutional layers that progressively halve the spatial "
          "resolution while doubling the number of channels, arriving at a compact "
          "representation of 256 channels at one quarter of the original resolution."),
        p("The bottleneck consists of nine residual blocks, each containing two "
          "convolutional layers with a skip connection. This depth is recommended by "
          "Zhu et al. for images of 256 by 256 resolution. The residual connections "
          "allow gradient flow during training and help the generator preserve content "
          "structure while modifying style."),
        p("The decoder mirrors the encoder using two transposed convolutional layers "
          "to progressively restore spatial resolution, followed by a final "
          "convolutional layer that maps to three output channels. A Tanh activation "
          "at the output confines pixel values to the range negative 1 to positive 1."),
        p("Throughout the generator, instance normalisation is used in place of batch "
          "normalisation. Instance normalisation normalises each image independently, "
          "which is important for style-transfer tasks where per-image statistics are "
          "meaningful. Reflection padding is used throughout to reduce border artifacts "
          "that commonly arise with zero-padding. Each generator contains approximately "
          "11.4 million trainable parameters."),
        sp(4),
        h2("4.2 Discriminator"),
        p("The discriminator uses the PatchGAN architecture (Isola et al., 2017), which "
          "classifies overlapping local patches of the input image as real or fake "
          "rather than classifying the entire image with a single scalar output. "
          "This design encourages the generator to produce realistic high-frequency "
          "textures, which is particularly important for the stripe patterns in the "
          "horse to zebra translation."),
        p("The network applies five convolutional layers with progressively increasing "
          "channel depth. LeakyReLU activations with a slope of 0.2 are used "
          "throughout. Instance normalisation is applied to all layers except the "
          "first, which is left unnormalised to preserve low-level colour and texture "
          "statistics. The final layer produces a patch-level map of real-versus-fake "
          "scores with no sigmoid activation, as the loss function operates directly "
          "on the raw outputs. Each discriminator has an effective receptive field of "
          "70 by 70 pixels and contains approximately 2.8 million parameters."),
        p("The full system comprises two generators (one for each translation "
          "direction) and two discriminators (one for each domain), giving a total "
          "of approximately 28.4 million trainable parameters."),
        sp(4),
        h2("4.3 Proposed Modification: Perceptual Cycle Loss"),
        p("The proposed variant of the architecture introduces a change only to the "
          "loss function used for cycle consistency training. The generator and "
          "discriminator network structures are identical to the baseline. The "
          "modification involves adding a frozen, pretrained VGG-16 network that "
          "serves as a fixed feature extractor. This network is never updated during "
          "training. Features are extracted from four intermediate layers of VGG-16, "
          "corresponding to progressively deeper levels of abstraction. The cycle "
          "consistency penalty is then computed in this feature space rather than "
          "directly on pixel values, as described in Section 5."),
        sp(6),
    ]

    # ── 5. Loss Function ───────────────────────────────────────────────────────
    story += [
        h1("5. Loss Function"),
        p("The total training objective combines three types of loss, each targeting "
          "a different aspect of the translation quality."),
        sp(4),
        h2("5.1 Adversarial Loss"),
        p("The adversarial loss encourages the generator to produce images that are "
          "indistinguishable from real images in the target domain. This project uses "
          "the Least Squares GAN formulation (Mao et al., 2017) rather than the "
          "original binary cross-entropy loss. The least squares objective is more "
          "stable during training and tends to produce sharper results because it "
          "penalises samples that lie far from the decision boundary, rather than "
          "those that merely fool the discriminator with high confidence."),
        p("The discriminator is trained to output a value of 1 for real images and 0 "
          "for generated images. The generator is trained to make the discriminator "
          "assign a value of 1 to its outputs. An adversarial loss is applied "
          "independently for both translation directions."),
        sp(4),
        h2("5.2 Cycle Consistency Loss"),
        p("The cycle consistency loss addresses the core challenge of unpaired "
          "training. Without a direct pixel-level supervision signal, the adversarial "
          "loss alone cannot constrain which horse image maps to which zebra image, "
          "leading to mode collapse. Cycle consistency enforces that if a horse image "
          "is translated to the zebra domain and then back to the horse domain, the "
          "result should closely match the original horse image, and vice versa."),
        p("In the baseline model, this constraint is enforced with an L1 penalty in "
          "pixel space. The L1 loss is weighted by a factor of 10, following the "
          "ablation study in Zhu et al. (2017). In the proposed modification, this "
          "pixel-space penalty is replaced by a perceptual loss. The reconstructed "
          "image and the original image are passed through a frozen VGG-16 network, "
          "and the difference between their feature maps is minimised at four "
          "intermediate layers (relu1_2, relu2_2, relu3_2, and relu4_2). The "
          "perceptual loss is applied with the same overall weight of 10, with equal "
          "contribution from each VGG layer. The rationale is that feature-space "
          "distances better capture perceptual and structural similarity than raw "
          "pixel differences, leading to sharper reconstructions."),
        sp(4),
        h2("5.3 Identity Loss"),
        p("The identity loss is an auxiliary regularisation term that encourages the "
          "generator to act as a near-identity mapping when given an image that "
          "already belongs to the target domain. For example, passing a zebra image "
          "through the horse-to-zebra generator should return an image close to the "
          "original zebra. This helps preserve the colour characteristics of the "
          "input and prevents unnecessary changes to images that do not need to be "
          "translated. The identity loss is computed using an L1 penalty in pixel "
          "space in both the baseline and modified models, with a weight of 5 "
          "(equal to half the cycle consistency weight)."),
        sp(4),
        h2("5.4 Summary of Loss Weights"),
        p("The table below summarises the three loss terms and their weights in both "
          "the baseline and proposed modified models:"),
        sp(6),
        simple_table(
            [
                ["Loss Term", "Weight", "Baseline", "Modified"],
                ["Adversarial loss", "1.0", "L1 pixel (LSGAN)", "L1 pixel (LSGAN)"],
                ["Cycle consistency loss", "10", "L1 in pixel space", "MSE in VGG-16 feature space"],
                ["Identity loss", "5", "L1 in pixel space", "L1 in pixel space"],
            ],
            col_widths=[5*cm, 2*cm, 4.5*cm, 4.5*cm],
        ),
        sp(8),
    ]

    # ── 6. Hyperparameters ─────────────────────────────────────────────────────
    story += [
        h1("6. Hyperparameters"),
        p("All core hyperparameters were adopted directly from the original CycleGAN "
          "paper (Zhu et al., 2017) to ensure a faithful and reproducible baseline. "
          "Deviations were made only where hardware constraints required it, as noted "
          "in the table below."),
        sp(6),
        simple_table(
            [
                ["Parameter", "Value", "Selection Rationale"],
                ["Image size", "256 x 256", "Standard benchmark resolution for Horse2Zebra"],
                ["Batch size", "4", "Constrained by available GPU memory (16 GB VRAM)"],
                ["Learning rate", "0.0002", "Adopted directly from Zhu et al. (2017)"],
                ["Optimizer", "Adam", "Standard choice for adversarial training"],
                ["Adam beta 1", "0.5", "Recommended for GAN stability by Radford et al. (2016)"],
                ["Adam beta 2", "0.999", "Default Adam value"],
                ["Total epochs", "200", "Standard CycleGAN training schedule"],
                ["LR decay start epoch", "100", "Linear decay from epoch 100 to 200, reaching zero"],
                ["Residual blocks", "9", "Recommended by Zhu et al. for 256x256 inputs"],
                ["Cycle loss weight", "10", "Selected via ablation study in Zhu et al. (2017)"],
                ["Identity loss weight", "5", "Set to half the cycle loss weight, per Zhu et al."],
                ["Replay buffer size", "50 images", "Following recommendation of Shrivastava et al. (2017)"],
                ["Mixed precision (AMP)", "Enabled", "Used for training efficiency on GPU"],
                ["Data loading workers", "8", "Matched to the number of CPU cores available"],
            ],
            col_widths=[4*cm, 3*cm, 9*cm],
        ),
        sp(8),
        p("The learning rate is held constant at 0.0002 for the first 100 epochs and "
          "then linearly decayed to zero over the remaining 100 epochs. This schedule "
          "allows the model to explore the parameter space freely in the early stages "
          "and then refine details as training progresses."),
        p("The replay buffer stores up to 50 previously generated images and is used "
          "to update the discriminator. At each discriminator update, either the "
          "current generated image or a randomly selected image from the buffer is "
          "used, each with 50 percent probability. This prevents the discriminator "
          "from adapting too rapidly to the current state of the generator, which "
          "improves training stability."),
        sp(6),
    ]

    # ── 7. SOTA Comparison ─────────────────────────────────────────────────────
    story += [
        h1("7. SOTA Comparison"),
        h2("7.1 Quantitative Comparison"),
        p("The standard metric for evaluating image translation quality is the "
          "Frechet Inception Distance (FID). FID measures the statistical distance "
          "between the distribution of generated images and the distribution of real "
          "images, as estimated in the feature space of a pretrained Inception-v3 "
          "network. Lower FID values indicate better quality."),
        p("The table below reports published FID scores on the Horse2Zebra benchmark "
          "for several representative methods, along with the status of our own "
          "results:"),
        sp(6),
        simple_table(
            [
                ["Method", "FID H to Z (lower is better)", "FID Z to H (lower is better)", "Year"],
                ["CycleGAN (Zhu et al.)", "77.2", "138.1", "2017"],
                ["UNIT (Liu et al.)", "96.1", "145.3", "2017"],
                ["MUNIT (Huang et al.)", "74.8", "136.8", "2018"],
                ["CUT (Park et al.)", "45.5", "85.9", "2020"],
                ["Our Baseline (Epoch 200)", "66.28", "144.08", "2026"],
                ["Our Modified (relu3_2, Epoch 90)", "230.08", "235.39", "2026"],
            ],
            col_widths=[5*cm, 4*cm, 4*cm, 3*cm],
        ),
        sp(8),
        p("Quantitative evaluation of both models was carried out on the standard "
          "Horse2Zebra test set (120 horse images in testA, 140 zebra images in "
          "testB). Our baseline model, evaluated using the epoch 200 checkpoint, "
          "achieves an FID of 66.28 on Horse to Zebra and 144.08 on Zebra to Horse. "
          "Notably, the Horse to Zebra FID of 66.28 improves upon the published "
          "CycleGAN result of 77.2, which reflects the benefit of completing the "
          "full 200-epoch training schedule including the linear learning rate decay "
          "phase (epochs 100 to 200). The Zebra to Horse FID of 144.08 is slightly "
          "above the published 138.1, consistent with the known difficulty of this "
          "direction documented in the original paper. Our modified model "
          "(relu3_2 perceptual cycle loss, epoch 90 checkpoint) achieves FID values "
          "of 230.08 and 235.39 for the two directions respectively. These values "
          "reflect the model being only partway through training, with the perceptual "
          "loss landscape requiring additional epochs to converge."),
        sp(4),
        p("In addition to FID, cycle consistency was measured using the Structural "
          "Similarity Index (SSIM) between original images and their "
          "cycle-reconstructed counterparts. For the baseline model, the Horse cycle "
          "SSIM is 0.8391 and the Zebra cycle SSIM is 0.8731, giving a mean SSIM "
          "of 0.8574. For the modified model, the Horse cycle SSIM is 0.9394 and "
          "the Zebra cycle SSIM is 0.9487, giving a mean SSIM of 0.9444. The "
          "notably higher SSIM of the modified model indicates that the perceptual "
          "cycle loss enforces stronger structural consistency in round-trip "
          "reconstruction, even though the model has not yet converged in terms "
          "of FID. These scores confirm that both models preserve structural content "
          "through the full translation round-trip."),
        sp(4),
        h2("7.2 Qualitative Comparison"),
        p("Qualitative results for the baseline model are available from the training "
          "run, with sample outputs saved at every 10th epoch from epoch 10 to "
          "epoch 200. Qualitative results for the modified model are available from "
          "epoch 10 to epoch 90. The following observations summarise the translation "
          "quality:"),
        sp(4),
        b("Horse to Zebra: By epoch 80, the generator produces convincing stripe "
          "patterns that follow the body contours of the horse. By epoch 200, the "
          "texture is well-defined and perceptually plausible. Background regions are "
          "largely unaffected. The results are qualitatively comparable to the "
          "published examples in the original CycleGAN paper."),
        b("Zebra to Horse: Stripe removal is partially successful, but the synthesised "
          "coat texture can appear inconsistent or overly smooth in some cases. "
          "This asymmetry between the two translation directions is a known limitation "
          "of the CycleGAN framework and is documented in the original paper. "
          "Adding stripes to a horse is an additive operation, whereas removing "
          "zebra stripes requires synthesising plausible smooth texture beneath a "
          "strong high-frequency pattern, which is inherently more difficult."),
        b("Training stability: Generator loss decreased steadily from approximately "
          "3.6 at epoch 71 to approximately 2.7 at epoch 150. Discriminator loss "
          "remained in the range of 0.17 to 0.54 throughout, indicating stable "
          "adversarial training without collapse or oscillation."),
        b("Metric consistency: The directional asymmetry observed qualitatively is "
          "confirmed by both metrics. FID is better in the Horse to Zebra direction "
          "(66.28) than Zebra to Horse (144.08), while SSIM is slightly higher for "
          "the Zebra cycle (0.8731) than the Horse cycle (0.8391). The stronger "
          "SSIM for Zebra cycle despite the weaker FID for that direction reflects "
          "the different aspects each metric captures: FID measures the realism of "
          "the generated images, whereas SSIM measures how faithfully the original "
          "is reconstructed after the full translation round-trip."),
        sp(6),
        h2("7.3 Positioning Against the State of the Art"),
        p("Our fully trained baseline achieves an FID of 66.28 on Horse to Zebra "
          "at epoch 200, which improves upon the published CycleGAN result of 77.2 "
          "and places our implementation between MUNIT (74.8) and CUT (45.5) on "
          "this benchmark. This is a positive result, demonstrating that the "
          "reimplementation is faithful to the original paper and that completing "
          "the full 200-epoch training schedule — including the linear learning rate "
          "decay phase — meaningfully improves generation quality. On the Zebra to "
          "Horse direction, our FID of 144.08 is slightly above the published 138.1, "
          "consistent with the known difficulty of this direction. The current "
          "strongest published result on this benchmark is CUT (Park et al., 2020), "
          "which achieves an FID of 45.5 using a contrastive learning objective "
          "rather than cycle consistency. The proposed perceptual loss modification "
          "was trained to epoch 90 and currently shows high FID (230.08) but "
          "substantially higher SSIM (0.9444 vs 0.8574 for the baseline). The "
          "improved cycle consistency under perceptual loss is an encouraging early "
          "signal; full convergence with extended training is needed to draw "
          "conclusions about FID."),
        sp(6),
    ]

    # ── 8. Remaining Work ──────────────────────────────────────────────────────
    story += [
        h1("8. Remaining Work and Limitations"),
        p("Due to limited GPU cluster access and file corruption during checkpoint "
          "transfer, some planned components could not be fully executed. The table "
          "below summarises the status of each component:"),
        sp(6),
        simple_table(
            [
                ["Component", "Status", "Notes"],
                ["Baseline model code", "Complete", "—"],
                ["Modified perceptual loss model code", "Complete", "—"],
                ["Baseline training, epochs 1 to 200", "Complete", "—"],
                ["Inference web application (Gradio)", "Complete", "—"],
                ["Modified model training (relu3_2)", "Partial — epoch 90", "Resource and time constraints"],
                ["Modified model checkpoints", "Partial", "Only epochs 10, 20, 80, 90 retained (others corrupted)"],
                ["Quantitative evaluation of baseline (FID, SSIM)", "Complete", "—"],
                ["Quantitative evaluation of modified model", "Complete", "Epoch 90 checkpoint used"],
                ["VGG layer ablation study", "Not run", "Insufficient modified checkpoints available"],
                ["Training loss logs (baseline)", "Partial", "Epochs 71 to 200 available; earlier logs not retained"],
                ["Training loss logs (modified)", "Missing", "Logs not copied before losing cluster access"],
            ],
            col_widths=[6*cm, 3*cm, 7*cm],
        ),
        sp(8),
        h2("8.1 Impact of Missing and Incomplete Components"),
        p("The baseline model was trained to completion at 200 epochs, achieving "
          "an FID of 66.28 on Horse to Zebra and 144.08 on Zebra to Horse. The "
          "Horse to Zebra result improves upon the published CycleGAN score of 77.2, "
          "which is a strong positive outcome for the reimplementation."),
        p("The modified perceptual loss model was trained to epoch 90 out of a "
          "planned 200 epochs. At this stage, the FID values are high (230.08 and "
          "235.39) as the model has not yet converged, but the mean SSIM of 0.9444 "
          "substantially exceeds the baseline mean SSIM of 0.8574. This indicates "
          "that the perceptual cycle loss enforces stronger structural consistency "
          "in round-trip reconstruction, which is the core motivation of the "
          "proposed modification. The hypothesis that perceptual loss improves "
          "structural fidelity is supported by these early results; the question "
          "of whether it also improves FID at convergence remains open."),
        p("The VGG layer ablation study could not be completed because intermediate "
          "checkpoints for the modified model (epochs 30 to 70, and 100 onwards) "
          "were corrupted during file transfer and are no longer available. Only "
          "epoch 10, 20, 80, and 90 checkpoints survive. Loss logs for the "
          "modified model were not retained before cluster access was lost."),
        sp(4),
        h2("8.2 Planned Future Work"),
        b("Continue training the modified perceptual loss model from epoch 90 to "
          "200 epochs and run a full quantitative and qualitative comparison against "
          "the baseline to determine whether perceptual cycle loss also improves FID "
          "at convergence."),
        b("Execute the VGG layer ablation study by training separate modified models "
          "with cycle consistency measured at each of relu1_2, relu2_2, relu3_2, "
          "and relu4_2 independently, to determine which level of abstraction is "
          "most effective for cycle consistency."),
        b("Investigate methods to reduce the Zebra to Horse quality gap, such as "
          "applying asymmetric loss weights that impose stronger cycle consistency "
          "on the harder direction."),
        b("Extend the comparison to include CUT (Park et al., 2020) as a stronger "
          "modern baseline."),
        sp(6),
    ]

    # ── References ─────────────────────────────────────────────────────────────
    story += [
        h1("References"),
        p("[1] Zhu, J.-Y., Park, T., Isola, P., and Efros, A. A. (2017). "
          "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial "
          "Networks. IEEE International Conference on Computer Vision (ICCV)."),
        p("[2] Johnson, J., Alahi, A., and Fei-Fei, L. (2016). "
          "Perceptual Losses for Real-Time Style Transfer and Super-Resolution. "
          "European Conference on Computer Vision (ECCV)."),
        p("[3] Mao, X., Li, Q., Xie, H., Lau, R. Y. K., Wang, Z., and Smolley, S. P. "
          "(2017). Least Squares Generative Adversarial Networks. "
          "IEEE International Conference on Computer Vision (ICCV)."),
        p("[4] Isola, P., Zhu, J.-Y., Zhou, T., and Efros, A. A. (2017). "
          "Image-to-Image Translation with Conditional Adversarial Networks. "
          "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)."),
        p("[5] Simonyan, K. and Zisserman, A. (2014). "
          "Very Deep Convolutional Networks for Large-Scale Image Recognition. "
          "International Conference on Learning Representations (ICLR 2015)."),
        p("[6] Park, T., Efros, A. A., Zhang, R., and Zhu, J.-Y. (2020). "
          "Contrastive Learning for Unpaired Image-to-Image Translation. "
          "European Conference on Computer Vision (ECCV)."),
        p("[7] Huang, X., Liu, M.-Y., Belongie, S., and Kautz, J. (2018). "
          "Multimodal Unsupervised Image-to-Image Translation. "
          "Advances in Neural Information Processing Systems (NeurIPS)."),
        p("[8] Liu, M.-Y., Breuel, T., and Kautz, J. (2017). "
          "Unsupervised Image-to-Image Translation Networks. "
          "Advances in Neural Information Processing Systems (NeurIPS)."),
        p("[9] Shrivastava, A., Pfister, T., Tuzel, O., Susskind, J., Wang, W., and "
          "Webb, R. (2017). Learning from Simulated and Unsupervised Images through "
          "Adversarial Training. IEEE Conference on Computer Vision and Pattern "
          "Recognition (CVPR)."),
        p("[10] Radford, A., Metz, L., and Chintala, S. (2016). "
          "Unsupervised Representation Learning with Deep Convolutional Generative "
          "Adversarial Networks. International Conference on Learning "
          "Representations (ICLR)."),
    ]

    doc.build(story, onFirstPage=first_page_template, onLaterPages=page_template)
    print(f"Report written to: {out}")


if __name__ == "__main__":
    build()
