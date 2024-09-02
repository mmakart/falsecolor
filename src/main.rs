use std::fs::File;
use std::io::BufWriter;
use std::iter::repeat_with;
use std::path::Path;

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
        let (x, y, p) = (smudge.x, smudge.y, smudge.p);

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
            d += a * b * delta * delta * delta * delta;
        }

        d
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
    p: Rgb,
}

impl Smudge {
    fn random(brushes: &[Rgb], width: u32, height: u32) -> Vec<Self> {
        let mut subsmudges = Vec::new();

        subsmudges.push(Self {
            x: rand::thread_rng().gen_range(0..width),
            y: rand::thread_rng().gen_range(0..height),
            p: brushes[rand::thread_rng().gen_range(0..brushes.len())],
        });

        if rand::thread_rng().gen_bool(0.5) {
            subsmudges.push(subsmudges[0].make_variation(brushes, width, height));
        }

        subsmudges
    }

    fn make_variation(&self, brushes: &[Rgb], width: u32, height: u32) -> Self {
        let xdelta = rand::thread_rng().gen_range(-1..1);
        let ydelta = rand::thread_rng().gen_range(-1..1);

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

fn score_smudge(smudges: Vec<Smudge>, canvas: &Image, target: &Image) -> (Vec<Smudge>, f32) {
    let mut im = canvas.clone();

    for smudge in &smudges {
        im.apply_smudge(*smudge);
    }

    (smudges, im.dist(target))
}

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: falsecolor <path/to/image.png>");
        return;
    };

    let brushes = vec![
        Rgb::from_u8(0xff, 0xff, 0xff), // white
        Rgb::from_u8(0xff, 0xf0, 0x00), // yellow
        Rgb::from_u8(0xff, 0x6c, 0x00), // orange
        Rgb::from_u8(0xff, 0x00, 0x00), // red
        Rgb::from_u8(0x8a, 0x00, 0xff), // violet
        Rgb::from_u8(0x00, 0x0c, 0xff), // blue
        Rgb::from_u8(0x0c, 0xff, 0x00), // green
        Rgb::from_u8(0xfc, 0x00, 0xff), // magenta
        Rgb::from_u8(0x00, 0xff, 0xea), // cyan
        Rgb::from_u8(0xbe, 0xbe, 0xbe), // grey
        Rgb::from_u8(0x7b, 0x7b, 0x7b), // dark_grey
        Rgb::from_u8(0x00, 0x00, 0x00), // black
        Rgb::from_u8(0x00, 0x64, 0x00), // dark_green
        Rgb::from_u8(0x96, 0x4b, 0x00), // brown
        Rgb::from_u8(0xff, 0xc0, 0xcb), // pink
    ];

    let target = Image::open(&path);

    let mut canvas = Image::blank_canvas(target.width, target.height);

    println!("initial dist={}", target.dist(&canvas));

    const STEPS: usize = 1024;
    const GENERATIONS: usize = 64;
    const INITIAL_SMUDGES: usize = 8192;
    const KEEP_ALIVE_AFTER_CULLING: usize = 24;
    const OFFSPRINGS: usize = 4;

    for _ in 0..STEPS {
        let mut smudges: Vec<_> =
            repeat_with(|| Smudge::random(&brushes, canvas.width, canvas.height))
                .take(INITIAL_SMUDGES)
                .map(|smudge| score_smudge(smudge, &canvas, &target))
                .collect();

        for _ in 0..GENERATIONS {
            smudges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            smudges.resize(KEEP_ALIVE_AFTER_CULLING, Default::default());

            let mut new_smudges = Vec::new();

            for smudge in &smudges {
                for _ in 0..OFFSPRINGS {
                    let smudge = smudge
                        .0
                        .iter()
                        .map(|s| s.make_variation(&brushes, canvas.width, canvas.height))
                        .collect();

                    new_smudges.push(score_smudge(smudge, &canvas, &target));
                }
            }

            new_smudges.extend_from_slice(&smudges);

            smudges = new_smudges;
        }

        smudges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let best_smudges = &smudges.first().unwrap().0;

        for sm in best_smudges {
            canvas.apply_smudge(*sm);
        }
    }

    canvas.save("out.png");

    println!("dist={}", target.dist(&canvas));
}
