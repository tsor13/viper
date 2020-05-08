// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     https://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// standard imports
#include <cfloat>
#include <fstream>
#include <iostream>
#include <thread>

#define TINYOBJLOADER_IMPLEMENTATION
#define USE_PNG

// include OpenGP and OpenGL
#include <OpenGP/GL/Application.h>
#include <OpenGP/GL/Components/GUICanvasComponent.h>
#include <OpenGP/GL/ImguiRenderer.h>
#include <OpenGP/Image/Image.h>

// include Octopus
// CHANGE
#include "OctopusComponent.h"
// TODO - do I need to modify Scene.h?
#include "Scene.h"

#include "CollisionGrid.cuh"

#define OPENGP_IMPLEMENT_ALL_IN_THIS_FILE
#include <OpenGP/util/implementations.h>

// namespace OpenGP
using namespace OpenGP;

// main
int main(int argc, char **argv) {

    // constant shadow size
    int shadow_size = 2048;

    // make OpenGP Application
    Application app;

    // TODO - From Scene.h? Or OpenGP?
    Scene scene;

    // make light
    auto &light_entity = scene.create_entity_with<CameraComponent>();
    light_entity.get<TransformComponent>().set_forward(
        Vec3(-1, -2, 0).normalized());
    light_entity.get<TransformComponent>().position = Vec3(50, 100, 0);

    Mat4x4 shadow_matrix =
        (light_entity.get_projection(shadow_size, shadow_size) *
         light_entity.get_view());

    // make floor
    auto &floor_entity = scene.create_entity_with<WorldRenderComponent>();
    auto &floor_renderer = floor_entity.set_renderer<SurfaceMeshRenderer>();
    floor_renderer.get_gpu_mesh().set_vpoint(
        {Vec3(-10000, 0, -10000), Vec3(10000, 0, -10000),
         Vec3(-10000, 0, 10000), Vec3(10000, 0, 10000)});
    floor_renderer.get_gpu_mesh().set_vnormal(
        {Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 1, 0)});
    floor_renderer.get_gpu_mesh().set_triangles({0, 1, 2, 1, 2, 3});

    // define floormat material
    Material floormat(R"GLSL(

        uniform sampler2D shadow_map;

        uniform vec3 light_pos;
        uniform mat4 shadow_matrix;
        uniform float shadow_near;
        uniform float shadow_far;

        vec3 world2uvdepth(vec3 pos, mat4 mat) {
            vec4 a = mat * vec4(pos, 1);
            vec3 b = a.xyz / a.w;
            return (b + vec3(1)) / 2;
        }

        float get_shadow_mask(vec2 uv) {
            return 1 - smoothstep(0.3, 0.5, length(uv - vec2(0.5, 0.5)));
        }

        vec3 get_ambient(vec3 pos) {
            vec3 ambient = vec3(0.14, 0.14, 0.18);

            vec3 uvd = world2uvdepth(pos, shadow_matrix);

            return ambient + vec3(0.2) * get_shadow_mask(uvd.xy);
        }

        float linear_shadow_depth(float d) {
            return shadow_near * shadow_far / (shadow_far + d * (shadow_near - shadow_far));
        }

        float get_shadow(vec3 pos) {
            ivec2 dim = textureSize(shadow_map, 0);
            vec3 uvd = world2uvdepth(pos, shadow_matrix);

            vec2 base_coord = uvd.xy * dim;
            ivec2 base_coord_i = ivec2(floor(base_coord));
            vec2 inter = fract(base_coord);

            mat4 shadow_depths;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    shadow_depths[i][j] = linear_shadow_depth(texelFetch(shadow_map, base_coord_i + ivec2(i-1, j-1), 0).r);
                }
            }

            float threshold = linear_shadow_depth(uvd.z) - 0.1;

            mat2 pcf_vals = mat2(0);
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    for (int x = 0; x < 3; ++x) {
                        for (int y = 0; y < 3; ++y) {
                            pcf_vals[i][j] += (shadow_depths[x + i][y + j] < threshold) ? 0 : (1.0 / 9.0);
                        }
                    }
                }
            }

            float a = mix(pcf_vals[0][0], pcf_vals[1][0], inter.x);
            float b = mix(pcf_vals[0][1], pcf_vals[1][1], inter.x);

            return mix(a, b, inter.y) * get_shadow_mask(uvd.xy);
        }

        vec4 fragment_shade() {
            vec3 pos = get_position();

            vec3 lightdir = normalize(light_pos - pos);

            vec3 white_color = vec3(1, 1, 1);
            vec3 black_color = vec3(0.6, 0.6, 0.6);

            vec3 background = (white_color + black_color) / 2;

            vec3 diffuse_color = white_color;

            vec3 modpos = mod(pos / 5, 1);

            if ((modpos.x < 0.5) ^^ (modpos.z < 0.5)) {
                diffuse_color = black_color;
            }

            float blur = exp(-2 * max(length(dFdx(pos)), length(dFdy(pos))));
            blur = clamp(2 * blur, 0, 1);

            diffuse_color = mix(background, diffuse_color, blur);

            vec3 ambient = get_ambient(pos);

            float shadow = get_shadow(pos);

            vec3 out_color = shadow * 0.85 * clamp(dot(get_normal(), normalize(lightdir)), 0, 1) * diffuse_color;

            out_color += ambient * diffuse_color;

            return vec4(out_color, 1);
        }

    )GLSL");
    floormat.set_property("ao_map", 6);
    floormat.set_property("shadow_map", 7);
    floormat.set_property("shadow_matrix", shadow_matrix);
    floormat.set_property("light_pos",
                          light_entity.get<TransformComponent>().position);
    floormat.set_property("shadow_near", light_entity.near_plane);
    floormat.set_property("shadow_far", light_entity.far_plane);
    floor_renderer.set_material(floormat);
    floor_renderer.rebuild();

    // make viper scene
    viper::Scene sim_scene;

    // Set OctopusComponent property to current scene
    OctopusComponent::v_scene = &sim_scene;
    // TODO - what is octoswarm??
    // CHANGE
    auto &octoswarm = scene.create_entity_with<OctopusComponent>();

    octoswarm.renderer->get_material().set_property("shadow_matrix",
                                                    shadow_matrix);
    octoswarm.renderer->get_material().set_property(
        "light_pos", light_entity.get<TransformComponent>().position);
    octoswarm.renderer->get_material().set_property("shadow_near",
                                                    light_entity.near_plane);
    octoswarm.renderer->get_material().set_property("shadow_far",
                                                    light_entity.far_plane);
    octoswarm.sphere_renderer->get_material().set_property("shadow_matrix",
                                                           shadow_matrix);
    octoswarm.sphere_renderer->get_material().set_property(
        "light_pos", light_entity.get<TransformComponent>().position);
    octoswarm.sphere_renderer->get_material().set_property(
        "shadow_near", light_entity.near_plane);
    octoswarm.sphere_renderer->get_material().set_property(
        "shadow_far", light_entity.far_plane);
    octoswarm.tsphere_renderer->get_material().set_property("shadow_matrix",
                                                            shadow_matrix);
    octoswarm.tsphere_renderer->get_material().set_property(
        "light_pos", light_entity.get<TransformComponent>().position);
    octoswarm.tsphere_renderer->get_material().set_property(
        "shadow_near", light_entity.near_plane);
    octoswarm.tsphere_renderer->get_material().set_property(
        "shadow_far", light_entity.far_plane);
    octoswarm.cannonball_renderer->get_material().set_property("shadow_matrix",
                                                               shadow_matrix);
    octoswarm.cannonball_renderer->get_material().set_property(
        "light_pos", light_entity.get<TransformComponent>().position);
    octoswarm.cannonball_renderer->get_material().set_property(
        "shadow_near", light_entity.near_plane);
    octoswarm.cannonball_renderer->get_material().set_property(
        "shadow_far", light_entity.far_plane);
    octoswarm.pillar_renderer->get_material().set_property("shadow_matrix",
                                                           shadow_matrix);
    octoswarm.pillar_renderer->get_material().set_property(
        "light_pos", light_entity.get<TransformComponent>().position);
    octoswarm.pillar_renderer->get_material().set_property(
        "shadow_near", light_entity.near_plane);
    octoswarm.pillar_renderer->get_material().set_property(
        "shadow_far", light_entity.far_plane);

    auto &c_entity = scene.create_entity_with<TrackballComponent>();
    c_entity.oriented = true;

    // constants - window?
    int ww = 3840, wh = 1080;

    // make framebuffer
    Framebuffer fb, fb_shadow;
    RGB8Texture color_map, color_map_shadow;
    D32FTexture depth_map, depth_map_shadow;

    auto realloc = [&](int w, int h) {
        color_map.allocate(w, h);
        depth_map.allocate(w, h);
    };

    realloc(ww, wh);

    depth_map_shadow.allocate(shadow_size, shadow_size);
    color_map_shadow.allocate(shadow_size, shadow_size);

    fb.attach_color_texture(color_map);
    fb.attach_depth_texture(depth_map);

    fb_shadow.attach_color_texture(color_map_shadow);
    fb_shadow.attach_depth_texture(depth_map_shadow);

    RGB8Texture colmap;
    Image<Eigen::Matrix<uint8_t, 3, 1>> colmap_cpu(2048, 2048);
    // Read in texture binary
    // TODO - what is it doing here?
    // CHANGE?
    std::ifstream("texture.bin", std::ios::binary).read(
        reinterpret_cast<char*>(&colmap_cpu(0, 0)), 12582912);
    colmap.upload(colmap_cpu);

    FullscreenQuad fsquad;

    // TODO - change defaults? Does this change the view?
    // No, change where says set_defaults
    bool show_pills = false;
    bool splitscreen = false;
    // bool splitscreen = true;

    // function to set pill visibility of octopi
    auto set_pill_visibility = [&](bool visible) {
        show_pills = visible;
        octoswarm.render_comp->visible = !visible;
        octoswarm.sphere_render_comp->visible = visible;
        octoswarm.vis_update();
    };

    // function to draw scene
    auto draw_scene = [&](int width, int height, int x, int y) {
        //======================================================================
        // Draw shadow map

        fb_shadow.bind();

        light_entity.draw(shadow_size, shadow_size);

        fb_shadow.unbind();

        //======================================================================
        // Draw scene with shadows

        fb.bind();

        glActiveTexture(GL_TEXTURE5);
        colmap.bind();
        glActiveTexture(GL_TEXTURE7);
        depth_map_shadow.bind();

        glActiveTexture(GL_TEXTURE0);

        // set up camera
        auto &cam = c_entity.get<CameraComponent>();

        cam.draw(color_map.get_width(), color_map.get_height(), 0, 0, false);

        // if sphere component is visible
        if (octoswarm.sphere_render_comp->visible) {
            RenderContext context;

            glDepthMask(GL_FALSE);

            context.aspect =
                (float)color_map.get_width() / (float)color_map.get_height();
            context.vfov = cam.vfov;
            context.near = cam.near_plane;
            context.far = cam.far_plane;
            context.eye = cam.get<TransformComponent>().position;
            context.forward = cam.get<TransformComponent>().forward();
            context.up = cam.get<TransformComponent>().up();

            context.update_view();
            context.update_projection();

            auto &renderable = *octoswarm.tsphere_render_comp;
            auto &transform = renderable.get<TransformComponent>();

            context.translation = transform.position;
            context.scale = transform.scale;
            context.rotation = transform.rotation;

            context.update_model();

            glEnable(GL_DEPTH_TEST);
            renderable.get_renderer().render(context);

            glDepthMask(GL_TRUE);
        }

        cam.draw_gui();

        fb.unbind();

        //======================================================================
        // Draw color map to window

        glViewport(x, y, width, height);
        fsquad.draw_texture(color_map);
    };

    // make window function
    auto &window = app.create_window([&](Window &window) {
        std::tie(ww, wh) = window.get_size();

        // TODO - what is this variable? splitscreen
        int fbw_new = splitscreen ? ww / 2 : ww;
        int fbh_new = wh;

        int fbw = color_map.get_width();
        int fbh = color_map.get_height();

        if (fbw_new != fbw || fbh_new != fbh) {
            realloc(fbw_new, fbh_new);
        }

        // if splitscreen, do half and half pill vilibility
        if (splitscreen) {
            // draw left half pills invisible
            set_pill_visibility(false);
            draw_scene(ww / 2, wh, 0, 0);
            // draw right half pills visible
            set_pill_visibility(true);
            draw_scene(ww / 2, wh, ww / 2, 0);
        } else {
            // if not split screen, call octoswarm.vis_update
            // TODO - what is vis_udpate?
            octoswarm.vis_update();
            draw_scene(ww, wh, 0, 0);
        }
    });

    // set window size and title
    window.set_size(ww, wh);
    window.set_title("VIPER Demo");

    // set up input
    // TODO - keyboard / mouse?
    auto &input = window.get_input();

    // TODO - camera position? Could be important
    // CHANGE
    c_entity.get<CameraComponent>().set_window(window);
    c_entity.center = Vec3(0, 1, 0);
    c_entity.get<TransformComponent>().position = Vec3(-12, 1, 0);

    // TODO - what is this
    auto &bsphere_entity = scene.create_entity_with<WorldRenderComponent>();
    auto &bsphere_renderer = bsphere_entity.set_renderer<SphereMeshRenderer>();

    // get mouse position
    auto get_mouse_ray = [&](Vec3 &eye, Vec3 &dir) {
        // get position
        Vec2 pos = input.mouse_position;
        pos[1] = wh - pos[1];

        int w = splitscreen ? ww / 2 : ww;
        pos = 2 * pos.cwiseQuotient(Vec2(w, wh)) - Vec2(1, 1);

        Vec4 cs(pos[0], pos[1], 0.1, 1);

        // TODO - what do? rotate?
        auto &cam = c_entity.get<CameraComponent>();
        Mat4x4 inv_mat = (cam.get_projection(w, wh) * cam.get_view()).inverse();

        Vec4 world = inv_mat * cs;
        Vec3 p = world.head<3>() / world[3];

        eye = c_entity.get<TransformComponent>().position;
        dir = (p - eye).normalized();
    };

    // variables to keep track of runtime performance
    int framerate = 0;
    double frametime = 0;
    double sim_frametime = 0;

    float playback = 1.0;

    int it_count = 10;

    // TODO - change?
    bool hide_gui = false;
    bool simulating = true;
    bool single_step = false;
    bool bsphere_vis = false;

    std::vector<float> framerates(120);

    // defaults for settings and physics
    auto set_defaults = [&]() {
        // TODO - change this to show pills??
        show_pills = false;
        // show_pills = true;
        octoswarm.render_comp->visible = !show_pills;
        octoswarm.sphere_render_comp->visible = show_pills;
        it_count = 10;
        sim_scene.gravity_strength = 1.0;
        playback = 1.0;
    };

    set_defaults();

    // Create GUI
    auto &canvas = scene.create_entity_with<GUICanvasComponent>();
    canvas.set_action([&]() {
        if (hide_gui)
            return;

        ImGui::SetNextWindowSize(ImVec2(400, 500));

        ImGui::Begin("Controls", nullptr,
                     ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoSavedSettings);

        char fr_label[256];
        sprintf(fr_label,
                "Framerate %i fps\n Total:   %3.1f ms\n Sim:     %3.1f ms",
                framerate, frametime, sim_frametime);

        ImGui::PlotLines(fr_label, &(framerates[0]), framerates.size(), 0, "",
                         0, 60);

        ImGui::Separator();

        if (ImGui::Button("Reset")) {
            octoswarm.reset();
        }

        ImGui::SameLine(0, 4);

        const char *bname = simulating ? "Pause" : "Resume";
        if (ImGui::Button(bname)) {
            simulating = !simulating;
        }

        if (!simulating) {
            ImGui::SameLine(0, 4);
            if (ImGui::Button("Step")) {
                single_step = true;
            }
        }

        ImGui::Checkbox("Split Screen", &splitscreen);

        if (ImGui::Checkbox("Show Primitives", &show_pills)) {
            set_pill_visibility(show_pills);
        }

        ImGui::SliderFloat("Gravity", &sim_scene.gravity_strength, -1.0f, 3.0f);
        ImGui::SliderInt("Solver Iterations", &it_count, 0, 50);

        if (ImGui::Button("Set Defaults")) {
            set_defaults();
            octoswarm.vis_update();
        }

        ImGui::Separator();

        const char *const scenes[] = {"Empty", "Pillars", "Cannonballs",
                                      "Explosion"};

        if (ImGui::ListBox("Scenes", &octoswarm.scene_index, scenes,
                           sizeof(scenes) / sizeof(scenes[0]))) {
            // TODO - resets octoswarm from beginning of simulation? could be important
            octoswarm.reset();
        }

        ImGui::Separator();

        ImGui::LabelText("Controls", "Look:                   Middle Mouse");
        ImGui::Text("Recenter:                Right Mouse");
        ImGui::Text("Pan:            Shift + Middle Mouse");
        ImGui::Text("Grab:                     Left Mouse");
        ImGui::Text("Shoot:                      Spacebar");
        ImGui::Text("Toggle Primitives:               F10");
        ImGui::Text("Pause/Resume:                    F11");
        ImGui::Text("Show/Hide Window:                F12");

        ImGui::End();
    });

    // place camera
    canvas.set_camera(c_entity.get<CameraComponent>());

    int chambered_cow = 0;

    long frame = 0;
    long sim_frame = 0;

    // keep track of statistics
    double last_time = glfwGetTime();
    double frame_avg = 0;
    double sim_frame_avg = 0;

    int held = 0;
    int selected = -1;

    // TODO - idk what this is 
    bool swapped_pills = false;
    bool swapped_pause = false;
    bool swapped_window = false;
    bool recentered = false;

    // app event listener
    app.add_listener<ApplicationUpdateEvent>(
        [&](const ApplicationUpdateEvent &) {
            SphereMesh temp_smesh;
            auto vs_temp =
                temp_smesh.add_vertex(viper::CollisionGrid::b_sphere);
            temp_smesh.add_sphere(vs_temp);
            bsphere_entity.visible = bsphere_vis;
            bsphere_renderer.upload_mesh(temp_smesh);

            // if mouseclick?
            if (input.get_mouse(0)) {
                Vec3 eye, dir;
                get_mouse_ray(eye, dir);

                // if just selected
                if (selected == -1) {
                    // TODO - if click on octopus, move it around?
                    selected = octoswarm.intersect(eye, dir);
                    // set velocity? to 0
                    sim_scene.state.xa[selected] = 0;
                // if octopus already selected
                } else {

                    Vec3 p = sim_scene.state.x[selected];

                    Vec3 x = p - eye;
                    Vec3 new_pos = p - (x - dir * dir.dot(x));
                    new_pos[1] =
                        std::max(new_pos[1], sim_scene.state.r[selected]);
                    sim_scene.state.x[selected] = new_pos;
                    sim_scene.state.xp[selected] = sim_scene.state.x[selected];
                }
            // unselect
            } else if (selected != -1) {
                sim_scene.state.xa[selected] = 1;
                selected = -1;
            }

            // TODO - idk what the get_mouse 1 is
            // move something around?
            if (input.get_mouse(1)) {
                Image<float> depth_im;
                depth_map.download(depth_im);

                int mxi = int(input.mouse_position[0]);
                int myi = int(wh - input.mouse_position[1]);

                auto &cam = c_entity.get<CameraComponent>();
                Mat4x4 inv_mat =
                    (cam.get_projection() * cam.get_view()).inverse();

                if (!recentered &&
                    !(mxi < 0 || mxi >= ww || myi < 0 || myi >= wh)) {

                    Vec3 uvdepth;
                    uvdepth.head<2>() = Vec2(float(mxi) / ww, float(myi) / wh);
                    uvdepth[2] = min(depth_im(myi, mxi), 0.999);

                    Vec4 dev(0, 0, 0, 1);
                    dev.head<3>() = 2 * uvdepth - Vec3::Ones();
                    Vec4 world_h = inv_mat * dev;

                    Vec3 new_center = world_h.head<3>() / world_h[3];
                    Vec3 dc = new_center - c_entity.center;
                    c_entity.center += dc;
                    c_entity.get<TransformComponent>().position += dc;

                    recentered = true;
                }
            } else {
                recentered = false;
            }

            // F10 changes pill visibility
            if (input.get_key(GLFW_KEY_F10)) {
                if (!swapped_pills) {
                    set_pill_visibility(!show_pills);
                    swapped_pills = true;
                }
            } else {
                swapped_pills = false;
            }
            // F11 pause/unpause
            if (input.get_key(GLFW_KEY_F11)) {
                if (!swapped_pause) {
                    simulating = !simulating;
                    swapped_pause = true;
                }
            } else {
                swapped_pause = false;
            }
            // F12 hides GUI
            if (input.get_key(GLFW_KEY_F12)) {
                if (!swapped_window) {
                    hide_gui = !hide_gui;
                    swapped_window = true;
                }
            } else {
                swapped_window = false;
            }

            // Space spits out new octopus
            if (input.get_key(GLFW_KEY_SPACE)) {
                // every 5 updates
                if ((held % 5) == 0) {
                    // TODO - important, makes new octopus
                    Vec3 p = c_entity.get<TransformComponent>().position;
                    Vec3 v = c_entity.get<TransformComponent>().forward();
                    octoswarm.set_position(chambered_cow, p + 3 * v, v);

                    chambered_cow = (chambered_cow + 1) % octoswarm.n_cows;
                }
                held++;
            } else {
                held = 0;
            }

            // keep track of frame statistics
            double frame_time = 0.0;
            double this_time = last_time;
            while (frame_time < 0.016667) {
                this_time = glfwGetTime();
                frame_time = this_time - last_time;
                std::this_thread::yield();
            }
            last_time = this_time;

            framerates.erase(framerates.begin());
            framerates.push_back(1.0 / frame_time);

            frame_avg += frame_time;

            if ((frame % 10) == 0) {
                frametime = 1000 * frame_avg / 10.0;
                framerate = 0.5 + 10.0 / frame_avg;
                frame_avg = 0;
            }

            // if simulating or supposed to take one step
            if (simulating || single_step) {
                // simulate one step forward
                // TODO - important
                double sim_time =
                    sim_scene.step(playback / 60.f, it_count, true);

                sim_frame_avg += sim_time;

                if ((sim_frame % 10) == 0) {
                    sim_frametime = sim_frame_avg / 10.0;
                    sim_frame_avg = 0;
                }

                single_step = false;

                sim_frame++;
            }

            scene.update();

            frame++;
        });

    app.run();

    return 0;
}
