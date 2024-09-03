use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::iter::repeat_with;
use std::path::Path;

use image::codecs::gif::{GifEncoder, Repeat};
use image::{Delay, Frame, ImageBuffer, Rgba};
use rand::Rng;

#[derive(Clone)]
struct Image {
    pixels: Vec<f32>,
    width: u32,
    height: u32,
}

impl Image {
    fn open(p: impl AsRef<Path>) -> Self {
        let decoder = png::Decoder::new(File::open(p).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        let bytes = &buf[..info.buffer_size()];

        assert!(info.bit_depth == png::BitDepth::Eight);

        let pixels: Vec<f32> = match info.color_type {
            png::ColorType::Rgb => bytes
                .iter()
                .map(|channel| (*channel as f32) / 255.0)
                .collect(),
            png::ColorType::Rgba => {
                let mut data = Vec::new();

                for pixel in bytes.chunks_exact(4) {
                    for channel in pixel.iter().take(3) {
                        data.push(*channel as f32 / 255.0);
                    }
                }

                data
            }
            ty => {
                println!("unknown color type: {ty:?}. only 24-bit RGB images are supported");
                unimplemented!()
            }
        };

        assert!(pixels.len() as u32 == info.width * info.height * 3);

        Self {
            pixels,
            width: info.width,
            height: info.height,
        }
    }

    fn save(&self, p: impl AsRef<Path>) {
        let pixels: Vec<_> = self
            .pixels
            .iter()
            .map(|channel| (channel * 255.0) as u8)
            .collect();

        let file = File::create(p).unwrap();
        let mut buf_writer = BufWriter::new(file);
        let mut encoder = png::Encoder::new(&mut buf_writer, self.width, self.height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);

        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&pixels).unwrap();
    }

    fn get_pixel(&self, x: u32, y: u32) -> Rgb {
        if x >= self.width || y >= self.height {
            return Rgb::default();
        }

        let offset = ((self.width * y + x) * 3) as usize;
        Rgb {
            r: self.pixels[offset],
            g: self.pixels[offset + 1],
            b: self.pixels[offset + 2],
        }
    }

    fn set_pixel(&mut self, x: u32, y: u32, p: Rgb) {
        if x >= self.width || y >= self.height {
            return;
        }

        let offset = ((self.width * y + x) * 3) as usize;

        self.pixels[offset] = p.r;
        self.pixels[offset + 1] = p.g;
        self.pixels[offset + 2] = p.b;
    }

    fn blend_pixel(&mut self, x: u32, y: u32, p: Rgb, alpha: f32) {
        let dst = self.get_pixel(x, y);

        let r = dst.r * (1.0 - alpha) + p.r * alpha;
        let g = dst.g * (1.0 - alpha) + p.g * alpha;
        let b = dst.b * (1.0 - alpha) + p.b * alpha;

        self.set_pixel(x, y, Rgb { r, g, b });
    }

    fn blank_canvas(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![1.0; 3 * width as usize * height as usize],
        }
    }

    fn apply_smudge(&mut self, smudge: Smudge) {
        let (x, y, p) = (smudge.x, smudge.y, smudge.p.rgb());

        self.blend_pixel(x, y, p, 0.7);
        self.blend_pixel(x, y - 1, p, 0.5);
        self.blend_pixel(x, y + 1, p, 0.5);
        self.blend_pixel(x - 1, y, p, 0.5);
        self.blend_pixel(x + 1, y, p, 0.5);
    }

    fn dist(&self, other: &Image) -> f32 {
        assert!(self.pixels.len() == other.pixels.len());

        let mut d = 0.0;

        for (a, b) in self.pixels.iter().zip(other.pixels.iter()) {
            // dot product with extreme punishment for bad color delta
            let delta = (a - b).abs();
            d += a * b * delta * delta * delta * delta * delta * delta * delta;
        }

        d
    }

    fn debug_image_buffer(&self) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        const SCALE: u32 = 8;
        let mut buf = ImageBuffer::new(self.width * SCALE, self.height * SCALE);
        for x in 0..self.width {
            for y in 0..self.height {
                let p = self.get_pixel(x, y);
                for ix in 0..SCALE {
                    for iy in 0..SCALE {
                        buf.put_pixel(
                            SCALE * x + ix,
                            SCALE * y + iy,
                            Rgba([
                                (p.r * 255.0) as u8,
                                (p.g * 255.0) as u8,
                                (p.b * 255.0) as u8,
                                0xff,
                            ]),
                        );
                    }
                }
            }
        }
        buf
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct Rgb {
    r: f32,
    g: f32,
    b: f32,
}

impl Rgb {
    fn from_u8(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct Smudge {
    x: u32,
    y: u32,
    p: Brush,
}

impl Smudge {
    fn make_variation(&self, brushes: &[Brush], width: u32, height: u32) -> Self {
        let xdelta = rand::thread_rng().gen_range(-1..=1);
        let ydelta = rand::thread_rng().gen_range(-1..=1);

        Self {
            x: (self.x as i32 + xdelta).clamp(0, width as i32 - 1) as u32,
            y: (self.y as i32 + ydelta).clamp(0, height as i32 - 1) as u32,
            p: if rand::thread_rng().gen_bool(0.9) {
                self.p
            } else {
                brushes[rand::thread_rng().gen_range(0..brushes.len())]
            },
        }
    }
}

#[derive(Clone, Default)]
struct SmudgePattern {
    smudges: Vec<Smudge>,
}

#[derive(Clone, Default)]
struct RankedSmudgePattern {
    pattern: SmudgePattern,
    rank: f32,
}

impl SmudgePattern {
    fn random(brushes: &[Brush], width: u32, height: u32) -> Self {
        let mut smudges = Vec::new();

        smudges.push(Smudge {
            x: rand::thread_rng().gen_range(0..width),
            y: rand::thread_rng().gen_range(0..height),
            p: brushes[rand::thread_rng().gen_range(0..brushes.len())],
        });

        if rand::thread_rng().gen_bool(0.5) {
            smudges.push(smudges[0].make_variation(brushes, width, height));
        }

        Self { smudges }
    }

    fn make_variation(&self, brushes: &[Brush], width: u32, height: u32) -> Self {
        Self {
            smudges: self
                .smudges
                .iter()
                .map(|s| s.make_variation(brushes, width, height))
                .collect(),
        }
    }

    fn rank(self, canvas: &Image, target: &Image) -> RankedSmudgePattern {
        let mut im = canvas.clone();

        for smudge in &self.smudges {
            im.apply_smudge(*smudge);
        }

        RankedSmudgePattern {
            pattern: self,
            rank: im.dist(target),
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
enum Brush {
    #[default]
    White,
    Yellow,
    Orange,
    Red,
    Violet,
    Blue,
    Green,
    Magenta,
    Cyan,
    Grey,
    DarkGrey,
    Black,
    DarkGreen,
    Brown,
    Pink,
}

impl Brush {
    #[rustfmt::skip]
    fn rgb(&self) -> Rgb {
        match self {
            Self::White     => Rgb::from_u8(0xff, 0xff, 0xff),
            Self::Yellow    => Rgb::from_u8(0xff, 0xf0, 0x00),
            Self::Orange    => Rgb::from_u8(0xff, 0x6c, 0x00),
            Self::Red       => Rgb::from_u8(0xff, 0x00, 0x00),
            Self::Violet    => Rgb::from_u8(0x8a, 0x00, 0xff),
            Self::Blue      => Rgb::from_u8(0x00, 0x0c, 0xff),
            Self::Green     => Rgb::from_u8(0x0c, 0xff, 0x00),
            Self::Magenta   => Rgb::from_u8(0xfc, 0x00, 0xff),
            Self::Cyan      => Rgb::from_u8(0x00, 0xff, 0xea),
            Self::Grey      => Rgb::from_u8(0xbe, 0xbe, 0xbe),
            Self::DarkGrey  => Rgb::from_u8(0x7b, 0x7b, 0x7b),
            Self::Black     => Rgb::from_u8(0x00, 0x00, 0x00),
            Self::DarkGreen => Rgb::from_u8(0x00, 0x64, 0x00),
            Self::Brown     => Rgb::from_u8(0x96, 0x4b, 0x00),
            Self::Pink      => Rgb::from_u8(0xff, 0xc0, 0xcb),
        }
    }
}

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        print_help()
    };

    let Some(out_path) = std::env::args().nth(2) else {
        print_help()
    };

    let steps = match std::env::args().nth(3) {
        Some(steps) => steps.parse::<usize>().unwrap_or_else(|_| print_help()),
        None => print_help(),
    };

    let target = Image::open(&path);
    let mut canvas = Image::blank_canvas(target.width, target.height);

    let generations = 64;
    let initial_smudges = 8192;
    let keep_alive_after_culling = 24;
    let offspring_count = 4;
    let linear_chunks = 32;

    let brushes = vec![
        Brush::White,
        Brush::Yellow,
        Brush::Orange,
        Brush::Red,
        Brush::Violet,
        Brush::Blue,
        Brush::Green,
        Brush::Magenta,
        Brush::Cyan,
        Brush::Grey,
        Brush::DarkGrey,
        Brush::Black,
        Brush::DarkGreen,
        Brush::Brown,
        Brush::Pink,
    ];

    eprintln!("initial dist={}", target.dist(&canvas));

    let gif_file = File::create(out_path.clone() + ".anim.gif").unwrap();
    let mut gif_encoder = GifEncoder::new(&gif_file);
    gif_encoder.set_repeat(Repeat::Infinite).unwrap();

    let mut smudge_history = Vec::new();

    for step in 0..steps {
        // Pick initial random smudges for this step
        let mut smudges: Vec<_> =
            repeat_with(|| SmudgePattern::random(&brushes, canvas.width, canvas.height))
                .take(initial_smudges)
                .map(|pattern| pattern.rank(&canvas, &target))
                .collect();

        // Find local optima by applying genetic algorithm to random smudges
        for _ in 0..generations {
            smudges.sort_by(|a, b| a.rank.partial_cmp(&b.rank).unwrap());
            smudges.resize(keep_alive_after_culling, Default::default());

            let mut new_smudges = Vec::new();

            for smudge in &smudges {
                for _ in 0..offspring_count {
                    let new_smudge = smudge
                        .pattern
                        .make_variation(&brushes, canvas.width, canvas.height)
                        .rank(&canvas, &target);

                    new_smudges.push(new_smudge);
                }
            }

            new_smudges.extend_from_slice(&smudges);

            smudges = new_smudges;
        }

        // Apply the best pattern found by genetic algorithm

        smudges.sort_by(|a, b| a.rank.partial_cmp(&b.rank).unwrap());

        let best_smudges = &smudges.first().unwrap().pattern.smudges;
        for smudge in best_smudges {
            canvas.apply_smudge(*smudge);
            smudge_history.push(*smudge);
        }

        // Linearize smudges in this chunk by sorting
        if step % linear_chunks == 0 && step != 0 {
            let mut new_canvas = Image::blank_canvas(canvas.width, canvas.height);

            smudge_history[step - 32..step].sort_by(|a, b| a.y.cmp(&b.y).then(a.x.cmp(&b.x)));

            for smudge in &smudge_history {
                new_canvas.apply_smudge(*smudge);
            }

            canvas = new_canvas;
        }
    }

    canvas.save(out_path.clone());

    // Write debug animation and instructions

    let mut new_canvas = Image::blank_canvas(canvas.width, canvas.height);
    let mut instructions = File::create(out_path.clone() + ".txt").unwrap();

    for smudges in smudge_history.chunks_exact(linear_chunks) {
        for smudge in smudges {
            gif_encoder
                .encode_frame(Frame::from_parts(
                    new_canvas.debug_image_buffer(),
                    0,
                    0,
                    Delay::from_numer_denom_ms(2, 100),
                ))
                .unwrap();

            writeln!(&mut instructions, "{:?} {}, {}", smudge.p, smudge.x + 1, smudge.y + 1).unwrap();
            new_canvas.apply_smudge(*smudge);
        }

        writeln!(&mut instructions, "---").unwrap();
    }

    gif_encoder
        .encode_frame(Frame::from_parts(
            new_canvas.debug_image_buffer(),
            0,
            0,
            Delay::from_numer_denom_ms(100, 1),
        ))
        .unwrap();

    eprintln!("final dist={}", target.dist(&canvas));
}

fn print_help() -> ! {
    eprintln!("usage: falsecolor <path/to/image.png> <path/to/output.png> <STEPS>");
    std::process::exit(2)
}
