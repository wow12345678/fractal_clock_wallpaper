use ::time::{Time, UtcDateTime, UtcOffset};
use std::f32::consts::TAU;
use std::ops::{Add, Mul};
use std::result::Result;
use std::sync::Arc;
use std::{f64, iter};
use winit::dpi::PhysicalSize;

use wgpu::util::DeviceExt;
use wgpu::{BindGroupEntry, BindGroupLayoutEntry, ExperimentalFeatures};

use winit::application::ApplicationHandler;
use winit::event::{KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LineData {
    start: [f32; 2],
    end: [f32; 2],
    color: [f32; 3],
    width: f32,
}

struct App {
    state: Option<State>,
}

impl App {
    pub fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes = Window::default_attributes().with_title("fractal_clock");

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        {
            self.state = Some(pollster::block_on(State::new(window)).unwrap());
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        {
            let state = match &mut self.state {
                Some(canvas) => canvas,
                None => return,
            };

            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => state.resize(size.width, size.height),
                WindowEvent::RedrawRequested => {
                    state.update();
                    match state.render() {
                        Ok(_) => {
                            state.window.request_redraw();
                        }
                        // Reconfigure the surface if it's lost or outdated
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let size = state.window.inner_size();
                            state.resize(size.width, size.height);
                        }
                        Err(e) => {
                            log::error!("Unable to render {}", e);
                        }
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => match (button, state.is_pressed())
                {
                    (MouseButton::Left, true) => {}
                    (MouseButton::Left, false) => {}
                    _ => {}
                },
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(code),
                            state: key_state,
                            ..
                        },
                    ..
                } => state.handle_key(event_loop, code, key_state.is_pressed()),
                _ => {}
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;

    fn mul(mut self, rhs: f32) -> Self::Output {
        self.x *= rhs;
        self.y *= rhs;
        self
    }
}

impl<T> From<[T; 2]> for Vec2
where
    T: Into<f32> + Copy,
{
    #[inline(always)]
    fn from(value: [T; 2]) -> Self {
        Self {
            x: value[0].into(),
            y: value[1].into(),
        }
    }
}

impl From<Vec2> for [f32; 2] {
    fn from(val: Vec2) -> Self {
        [val.x, val.y]
    }
}

impl Vec2 {
    fn rot(&self, other: Vec2) -> Vec2 {
        Self {
            x: self.y * other.x - self.x * other.y,
            y: self.x * other.x + self.y * other.y,
        }
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

struct Clock {
    time: f64,
    last_calc_time: f64,
    start_line_width: f32,
    depth: usize,
    length_factor: f32,
    luminance_factor: f32,
    width_factor: f32,
    line_count: usize,
    phys_size: PhysicalSize<u32>,
    zoom: f32,
    line_cache: Vec<LineData>,
}

impl Clock {
    fn update_time(&mut self) {
        let dur = UtcDateTime::now()
            .to_offset(UtcOffset::from_hms(2, 0, 0).expect("has to be valid offset"))
            .time()
            .duration_since(Time::MIDNIGHT)
            .as_seconds_f64();
        self.time = dur;
    }

    fn calc_line_data(&mut self) -> Vec<LineData> {
        struct Hand {
            length: f32,
            angle: f32,
            vec: Vec2,
        }

        impl Hand {
            fn from_length_angle(length: f32, angle: f32) -> Self {
                let angled: Vec2 = [angle.cos(), -angle.sin()].into();
                Self {
                    length,
                    angle,
                    vec: angled * length,
                }
            }
        }

        let angle_from_period =
            |period| TAU * (self.time.rem_euclid(period) / period) as f32 - TAU / 4.0;

        let color_from_luminance = |lum: u8| {
            let l = lum as f64 / 255.0;
            wgpu::Color {
                r: l,
                g: l,
                b: l,
                a: 1.0,
            }
        };

        let aspect_ratio = self.phys_size.width as f32 / self.phys_size.height as f32;

        let hands = [
            // Second hand:
            Hand::from_length_angle(self.length_factor, angle_from_period(60.0)),
            // Minute hand:
            Hand::from_length_angle(self.length_factor, angle_from_period(60.0 * 60.0)),
            // Hour hand:
            Hand::from_length_angle(0.5, angle_from_period(12.0 * 60.0 * 60.0)),
        ];

        let max_lines = 1 + 2_usize.pow(self.depth as u32 + 2_u32);
        let mut lines: Vec<LineData> = Vec::with_capacity(max_lines);

        let mut paint_line = |points: [Vec2; 2], color: wgpu::Color, width: f32| {
            let transform_coords = |mut pos: Vec2| -> Vec2 {
                pos.x *= self.zoom;
                pos.y *= self.zoom;

                if aspect_ratio > 1.0 {
                    pos.x /= aspect_ratio;
                } else {
                    pos.y *= aspect_ratio;
                }
                pos
            };

            let c = [color.r as f32, color.g as f32, color.b as f32];

            lines.push(LineData {
                start: transform_coords(points[0]).into(),
                end: transform_coords(points[1]).into(),
                color: c,
                width,
            });
        };

        let hand_rotations = [
            hands[0].angle - hands[2].angle + TAU / 2.0,
            hands[1].angle - hands[2].angle + TAU / 2.0,
        ];

        let hand_rotors: [Vec2; 2] = [
            [
                hands[0].length * hand_rotations[0].sin(),
                hands[0].length * hand_rotations[0].cos(),
            ]
            .into(),
            [
                hands[1].length * hand_rotations[1].sin(),
                hands[1].length * hand_rotations[1].cos(),
            ]
            .into(),
        ];

        #[derive(Clone, Copy)]
        struct Node {
            pos: Vec2,
            dir: Vec2,
        }

        let mut nodes = Vec::new();

        let mut width = self.start_line_width;

        for (i, hand) in hands.iter().enumerate() {
            let center: Vec2 = [0.0, 0.0].into();
            let end = center + hand.vec;
            paint_line([center, end], color_from_luminance(255), width);
            if i < 2 {
                nodes.push(Node {
                    pos: end,
                    dir: hand.vec,
                });
            }
        }

        let mut luminance = 0.7; // Start dimmer than main hands

        let mut new_nodes = Vec::new();
        for _ in 0..self.depth {
            new_nodes.clear();
            new_nodes.reserve(nodes.len() * 2);

            luminance *= self.luminance_factor;
            width *= self.width_factor;

            let luminance_u8 = (255.0 * luminance).round() as u8;
            if luminance_u8 == 0 {
                break;
            }

            for &rotor in &hand_rotors {
                for a in &nodes {
                    let new_dir = rotor.rot(a.dir);
                    let b = Node {
                        pos: a.pos + new_dir,
                        dir: new_dir,
                    };
                    paint_line([a.pos, b.pos], color_from_luminance(luminance_u8), width);
                    new_nodes.push(b);
                }
            }

            std::mem::swap(&mut nodes, &mut new_nodes);
        }
        self.line_count = lines.len();

        lines
    }
}

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    line_data_buf: wgpu::Buffer,
    line_data_bind_group: wgpu::BindGroup,
    window: Arc<Window>,
    clock: Clock,
}

impl State {
    async fn new(window: Arc<Window>) -> anyhow::Result<State> {
        let dur = UtcDateTime::now()
            .to_offset(UtcOffset::from_hms(2, 0, 0).expect("has to be valid offset"))
            .time()
            .duration_since(Time::MIDNIGHT)
            .as_seconds_f64();

        let size = window.inner_size();

        let mut clock = Clock {
            time: dur,
            last_calc_time: dur,
            start_line_width: 0.002,
            depth: 9,
            length_factor: 0.8,
            luminance_factor: 0.60,
            width_factor: 0.8,
            line_count: 0,
            phys_size: size,
            zoom: 0.40,
            line_cache: Vec::new(),
        };

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off, // Trace path
                experimental_features: ExperimentalFeatures::disabled(),
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an Srgb surface texture. Using a different
        // one will result all the colors comming out darker. If you want to support non
        // Srgb surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let lines = clock.calc_line_data();
        let line_data_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("line_data_buf"),
            contents: bytemuck::cast_slice(&lines),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let line_buf_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("line_data_buf_layout"),
        });

        let line_data_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("line_buf_bind_group"),
            layout: &line_buf_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: line_data_buf.as_entire_binding(),
            }],
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&line_buf_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
            // Useful for optimizing shader compilation on Android
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            line_data_buf,
            line_data_bind_group,
            is_surface_configured: false,
            window,
            clock,
        })
    }

    fn update(&mut self) {
        self.clock.update_time();
        self.clock.phys_size = self.window.inner_size();

        // recalc every 16ms for 60fps
        if (self.clock.time - self.clock.last_calc_time).abs() > 0.016 {
            self.clock.line_cache = self.clock.calc_line_data();
            self.clock.last_calc_time = self.clock.time;

            self.queue.write_buffer(
                &self.line_data_buf,
                0,
                bytemuck::cast_slice(&self.clock.line_cache),
            );
        }
    }

    fn render(&self) -> Result<(), wgpu::SurfaceError> {
        // We can't render unless the surface is configured
        if !self.is_surface_configured {
            println!("Surface not configured");
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.line_data_bind_group, &[]);
            render_pass.draw(0..6, 0..self.clock.line_count as u32);
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
        }
    }

    fn handle_key(
        &self,
        event_loop: &ActiveEventLoop,
        key: winit::keyboard::KeyCode,
        pressed: bool,
    ) {
        match (key, pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            (KeyCode::Space, true) => {}
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
