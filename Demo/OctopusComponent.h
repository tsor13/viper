// Lcensed under the Apache License, Version 2.0 (the "License");
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

#pragma once

// include standard libraries
#include <cfloat>

// Load OpenGP
#include <OpenGP/GL/Components/TrackballComponent.h>
#include <OpenGP/GL/Components/WorldRenderComponent.h>
#include <OpenGP/GL/Scene.h>
#include <OpenGP/SphereMesh/GL/SphereMeshRenderer.h>
#include <OpenGP/SurfaceMesh/GL/SurfaceMeshRenderer.h>

#include "tiny_obj_loader.h"

// Load Demo file
#include "Octopus.h"
// Load VIPER files
#include "Scene.h"
#include "Subprocess.h"
#include "Viper_json.h"

// I'm assuming for drawing pills?
#define VIPER_TEXTURE 0
#define VIPER_GREY 1
#define VIPER_GOLD 2
#define VIPER_CLEAR_RED 3

// make new namespace
namespace OpenGP {
// OctopusData data structure
struct OctopusData {
    // TODO - what are these constraints? Are they resting states?
    std::vector<int> radius_constraints;
    std::vector<int> volume_constraints;
    std::vector<int> stretch_constraints;
    std::vector<int> distance_constraints;
    std::vector<int> bend_constraints;
    std::vector<int> cannonball_constraints;


    // set whether each constraint is enabled in scene
    void set_enabled(bool enabled, viper::Scene &scene) {
        for (auto i : radius_constraints)
            scene.constraints.radius[i].enabled = enabled;
        for (auto i : volume_constraints)
            scene.constraints.volume[i].enabled = enabled;
        for (auto i : stretch_constraints)
            scene.constraints.stretch[i].enabled = enabled;
        for (auto i : distance_constraints)
            scene.constraints.distance[i].enabled = enabled;
        for (auto i : bend_constraints)
            scene.constraints.bend[i].enabled = enabled;
    }

    // TODO - whaht is cannonball?
    // set whether cannonball enabled in scene
    void set_cannonball_enabled(bool enabled, viper::Scene &scene) {
        for (auto i : cannonball_constraints)
            scene.constraints.distance[i].enabled = enabled;
    }
};

// TODO - where does Component come from? OpenGP?
// OctopusComponent class
class OctopusComponent : public Component {
  public:
  // scene where present
    static viper::Scene *v_scene;

    // TODO - test this
    // num of octopi to simulate
    const int n_cows = 1;
    int n_active = 90;

    // which scene - normal, pillars, etc.
    // 0 - normal
    // 1 - pillars
    // 2 - cannonballs
    // 3 - explode from origin
    // 4 - mine
    int scene_index = 0;

    int n_cube = 3;

    // what is active cannonball?
    bool cannonballs_active = false;

    // renderers
    WorldRenderComponent *render_comp;
    SurfaceMeshRenderer *renderer;
    WorldRenderComponent *sphere_render_comp;
    SphereMeshRenderer *sphere_renderer;
    WorldRenderComponent *tsphere_render_comp;
    SphereMeshRenderer *tsphere_renderer;
    WorldRenderComponent *cannonball_render_comp;
    SphereMeshRenderer *cannonball_renderer;
    WorldRenderComponent *pillar_render_comp;
    SphereMeshRenderer *pillar_renderer;

    // tentacle
    std::map<int,float> original_distances;
    std::vector<std::vector<int>> tentacle_groups;

    // ids
    std::vector<std::vector<int>> v_ids, p_ids;
    std::vector<int> cannonball_ids;
    std::vector<int> pillar_ids;
    // vector of OctopusData struct
    std::vector<OctopusData> cow_data;

    // high def surface mesh
    SurfaceMesh mesh;
    // smesh keeps track of viper primitives, idk cannonball_smesh
    OpenGP::SphereMesh smesh, cannonball_smesh;

    // spheres and pills
    std::vector<Vec4> spheres;
    std::vector<Vec2i> all_pills;
    // TODO - what is a control pill?
    std::vector<int> control_pills;
    std::vector<float> compliances;
    std::vector<float> masses;

    // I believe for moving the camera around
    std::vector<std::vector<Mat4x4>> init_transforms;

    // since we're using Vertex a lot, I assum
    using V = OpenGP::SphereMesh::Vertex;

    // initialize
    void init() {
        render_comp = &(require<WorldRenderComponent>());
        renderer = &(render_comp->set_renderer<SurfaceMeshRenderer>());

        // define Octomat material
        // TODO - important?
        Material octomat(R"GLSL(

            flat out int gid;

            void vertex_shade() {

                gid = gl_InstanceID;

            }

        )GLSL",
                        R"GLSL(

            flat in int gid[];
            flat out int fid;

            void geometry_vertex_shade(int v) {
                fid = gid[v];
            }

        )GLSL",
                        R"GLSL(

            uniform sampler2D diffuse;
            uniform sampler2D shadow_map;
            uniform int material;

            uniform vec3 light_pos;
            uniform mat4 shadow_matrix;
            uniform float shadow_near;
            uniform float shadow_far;

            flat in int fid;

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

                float threshold = linear_shadow_depth(uvd.z) - 0.4;

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

            // http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl

            vec3 rgb2hsv(vec3 c) {
                vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

                float d = q.x - min(q.w, q.y);
                float e = 1.0e-10;
                return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
            }

            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }

            vec4 fragment_shade() {

                vec3 pos = get_position();

                vec3 lightdir = normalize(light_pos - pos);

                vec3 colors[7];
                colors[0] = vec3(1.00f, 0.98f, 0.28f);
                colors[1] = vec3(0.00f, 0.68f, 0.49f);
                colors[2] = vec3(0.14f, 0.49f, 0.76f);
                colors[3] = vec3(0.92f, 0.40f, 0.00f);
                colors[4] = vec3(0.00f, 0.80f, 1.00f);
                colors[5] = vec3(0.89f, 0.52f, 0.71f);
                colors[6] = vec3(1.00f, 0.68f, 0.00f);

                float spec_power = 80;
                vec3 diffuse_color = vec3(1, 1, 1);
                float alpha = 1.0;
                if (material == 0) {
                    vec3 base_color = colors[fid % 7];

                    vec2 uv = get_texcoord();
                    diffuse_color = texture(diffuse, vec2(uv.x, 1 - uv.y)).rgb;

                    vec3 diffuse_hsv = rgb2hsv(diffuse_color);
                    diffuse_hsv.r = rgb2hsv(base_color).r;
                    diffuse_color = hsv2rgb(diffuse_hsv);
                }
                if (material == 2) {
                    diffuse_color = vec3(220, 200, 120) / 255.0;
                    spec_power = 200;
                }
                if (material == 3) {
                    diffuse_color = vec3(220, 60, 60) / 255.0;
                    alpha = 0.5;
                }

                vec3 ambient = get_ambient(pos);
                ambient *= (1 + abs(dot(get_normal(), get_forward()))) / 2;

                float shadow = get_shadow(pos);

                vec3 out_color = shadow * 0.85 * clamp(dot(get_normal(), normalize(lightdir)), 0, 1) * diffuse_color;
                out_color += vec3(1) * shadow * pow(clamp(dot(get_forward(), reflect(lightdir, get_normal())), 0, 1), spec_power);

                out_color += ambient * diffuse_color;

                return vec4(out_color, alpha);
            }

        )GLSL");
        // set properties (defined above?)
        octomat.set_property("material", VIPER_TEXTURE);
        octomat.set_property("diffuse", 5);
        octomat.set_property("ao_map", 6);
        octomat.set_property("shadow_map", 7);

        renderer->set_material(octomat);
        renderer->rebuild();
        render_comp->visible = true;

        // have spheres render to above specifications
        sphere_render_comp =
            &(get_scene().create_entity_with<WorldRenderComponent>());
        sphere_renderer =
            &(sphere_render_comp->set_renderer<SphereMeshRenderer>());

        sphere_renderer->set_material(octomat);
        sphere_renderer->get_material().set_property("material", VIPER_GOLD);
        sphere_renderer->rebuild();
        sphere_render_comp->visible = false;

        // have pills render to above specs
        tsphere_render_comp =
            &(get_scene().create_entity_with<WorldRenderComponent>());
        tsphere_renderer =
            &(tsphere_render_comp->set_renderer<SphereMeshRenderer>());

        tsphere_renderer->set_material(octomat);
        tsphere_renderer->no_spheres = true;
        tsphere_renderer->get_material().set_property("material",
                                                      VIPER_CLEAR_RED);
        tsphere_renderer->rebuild();
        tsphere_render_comp->visible = false;

        // render cannonbals? grey?
        cannonball_render_comp =
            &(get_scene().create_entity_with<WorldRenderComponent>());
        cannonball_renderer =
            &(cannonball_render_comp->set_renderer<SphereMeshRenderer>());

        cannonball_renderer->set_material(octomat);
        cannonball_renderer->get_material().set_property("material",
                                                         VIPER_GREY);
        cannonball_renderer->rebuild();

        // render pillars
        pillar_render_comp =
            &(get_scene().create_entity_with<WorldRenderComponent>());
        pillar_renderer =
            &(pillar_render_comp->set_renderer<SphereMeshRenderer>());

        pillar_renderer->set_material(octomat);
        pillar_renderer->get_material().set_property("material", VIPER_GREY);
        pillar_renderer->rebuild();

        // make pillars
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {

                float r = 0.5;
                Vec3 c0 = Vec3(4 * (i - 2), 0, 4 * (j - 2));
                Vec3 c1 = Vec3(4 * (i - 2), 4, 4 * (j - 2));

                int id0 = v_scene->addParticle(c0, r, 1.0, true);
                int id1 = v_scene->addParticle(c1, r, 1.0, true);

                pillar_ids.push_back(id0);
                pillar_ids.push_back(id1);

                v_scene->addPill(id0, id1, true);
            }
        }

        // read in SurfaceMesh for octopus
        {
            SurfaceMesh tempmesh;
            // WINGATE
            // TOGGLE THIS TO READ TENTACLE OR OCTOPUS
            bool read_tentacle = false;
            if (read_tentacle){ // READ IN TENTACLE OBJ
                bool success = tempmesh.read("Tentacle_convert.obj");
                std::cout << "Read tentacle exited with code: " << success << std::endl;
                auto vpindex = tempmesh.add_vertex_property<int>("v:pindex");

                unsigned i = 0;
                // for each vertex, add vertex properties to tempmesh
                for (auto vert : tempmesh.vertices()){
                    vpindex[vert] = i;
                    i++;
                }
            } else { // READ IN OCTOPUS
                // read in from mesh.bin
                std::ifstream istream("mesh.bin", std::ios::binary);

                // 24842 Vec3 point vectors
                std::vector<Vec3> pointvec(24842);
                istream.read(reinterpret_cast<char*>(&pointvec[0]), 24842 * sizeof(Vec3));
                
                // 24842 Vec3 normal vectors
                std::vector<Vec3> normalvec(24842);
                istream.read(reinterpret_cast<char*>(&normalvec[0]), 24842 * sizeof(Vec3));
                
                // 24842 Vec2 texture vectors
                std::vector<Vec2> texvec(24842);
                istream.read(reinterpret_cast<char*>(&texvec[0]), 24842 * sizeof(Vec2));
                
                // 24842 int p indices
                std::vector<int> pidxvec(24842);
                istream.read(reinterpret_cast<char*>(&pidxvec[0]), 24842 * sizeof(int));
                
                // 12312 triangle faces
                std::vector<int> trivec(12312 * 3);
                istream.read(reinterpret_cast<char*>(&trivec[0]), 12312 * 3 * sizeof(int));

                // for each point, add vertex to tempmesh
                for (int i = 0; i < 24842; ++i) {
                    auto vert = tempmesh.add_vertex(pointvec[i]);
                }

                // Vertex Properties
                // Vec2 texture coordinates
                auto vtexcoord = tempmesh.add_vertex_property<Vec2>("v:texcoord");
                // Vec3 normal vectors
                auto vnormal = tempmesh.add_vertex_property<Vec3>("v:normal");
                // int p index
                auto vpindex = tempmesh.add_vertex_property<int>("v:pindex");

                // for each vertex, add vertex properties to tempmesh
                for (int i = 0; i < 24842; ++i) {
                    SurfaceMesh::Vertex vert(i);
                    vnormal[vert] = normalvec[i];
                    vtexcoord[vert] = texvec[i];
                    vpindex[vert] = pidxvec[i];
                }
                // for each face, add to tempmesh
                for (int i = 0; i < 12312; ++i) {
                    tempmesh.add_triangle(SurfaceMesh::Vertex(trivec[3*i]),
                                    SurfaceMesh::Vertex(trivec[3*i + 1]),
                                    SurfaceMesh::Vertex(trivec[3*i + 2]));
                }

            }
            // set mesh to tempmesh
            mesh = tempmesh;
        }

        // make an octopus from VIPER primitives as defined it Octopus.h
        cow::get_octopus(spheres, all_pills, control_pills, masses,
                         compliances);

        // for each sphere, add vertex to smesh
        for (int i = 0; i < spheres.size(); ++i) {
            auto &v = spheres[i];
            smesh.add_vertex(v);
        }

        // for each control pill, add edge to smesh
        for (int i : control_pills) {
            auto pill = all_pills[i];
            smesh.add_edge(V(pill[0]), V(pill[1]));
        }

        // output string
        std::string output;

        // get vertex p indices from mesh
        auto vpindex = mesh.get_vertex_property<int>("v:pindex");
        // int to int map
        // maps p indices to 0, 1, 2, ...
        std::map<int, int> ind_verts;
        {
            //
            // build watertight mesh
            //
            // store positions of vertices
            std::vector<Vec3> verts;
            // for each vertex in mesh
            for (auto vert : mesh.vertices()) {
                // int = vector p index from mesh
                int ind = vpindex[vert];
                // if not in ind_verts?
                if (ind_verts.find(ind) == ind_verts.end()) {
                    // counts up 0, 1, ...
                    ind_verts[ind] = verts.size();
                    // store position of vertex
                    verts.push_back(mesh.position(vert));
                }
            }

            // surface mesh faces
            std::vector<Vec3i> tris;
            // for each face in mesh
            for (auto face : mesh.faces()) {
                Vec3i tri;
                int i = 0;
                //for each vertex in face
                for (auto vert : mesh.vertices(face))
                    // next entry in tri is vertex index (0, 1, ...)
                    tri[i++] = ind_verts[vpindex[vert]];
                // save face in tris
                tris.push_back(tri);
            }
            //
            // compute weights
            //
            rapidjson::MemoryPoolAllocator<> alloc;
            rapidjson::Document j(&alloc);
            j.SetObject();
            rapidjson::Document::AllocatorType &allocator = j.GetAllocator();

            // send surface mesh vertices
            j.AddMember("vertices", viper::to_json(verts, allocator),
                        allocator);
            // send surface mesh triangles
            j.AddMember("triangles", viper::to_json(tris, allocator),
                        allocator);

            rapidjson::Value spheres_array(rapidjson::kArrayType);
            rapidjson::Value pills_array(rapidjson::kArrayType);

            // send VIPER spheres
            j.AddMember("spheres",
                        viper::to_json(
                            smesh.get_vertex_property<Vec4>("v:point").vector(),
                            allocator),
                        allocator);
            // send VIPER pills
            j.AddMember(
                "pills",
                viper::to_json(
                    smesh.get_edge_property<Vec2i>("e:connectivity").vector(),
                    allocator),
                allocator);

            // json accept output
            rapidjson::StringBuffer strbuf;
            strbuf.Clear();
            rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
            j.Accept(writer);

            // run Sphereweights, which calculates weights to mesh I'm assuming
            int proc_return =
                viper::run_process("sphereweights", strbuf.GetString(), output);
            std::cout << "Sphereweights exited with code: " << proc_return
                      << std::endl;
        }

        // parse the json
        {
            rapidjson::Document j;
            j.Parse(output.c_str());

            // read in weights (floats matrix)
            auto all_weights =
                viper::from_json<std::vector<std::vector<float>>>(j["weights"]);
            // read in bone ids (ints)
            auto all_bones =
                viper::from_json<std::vector<std::vector<int>>>(j["bone_ids"]);

            std::vector<Vec4> weights(all_weights.size());
            std::vector<Vec4i> bones(all_bones.size());

            // iterate thruogh weights
            for (int i = 0; i < weights.size(); ++i) {
                // make room in allweights for 4
                while (all_weights[i].size() < 4) {
                    all_weights[i].push_back(0.f);
                    all_bones[i].push_back(0);
                }
                // assign to weights
                weights[i] = Vec4(all_weights[i][0], all_weights[i][1],
                                  all_weights[i][2], all_weights[i][3]);
                // normalize
                float sum = weights[i].sum();
                weights[i] /= sum;
                // set bone ids
                bones[i] = Vec4i(all_bones[i][0], all_bones[i][1],
                                 all_bones[i][2], all_bones[i][3]);
            }

            // add vertex properties
            // skin weight Vec4
            auto weights_prop = mesh.add_vertex_property<Vec4>("v:skinweight");
            // bone ids Vec4i
            auto bone_ids_prop = mesh.add_vertex_property<Vec4i>("v:boneid");

            // for each vertex in mesh
            for (auto vert : mesh.vertices()) {
                // set mesh weights to corresponding weights
                weights_prop[vert] = weights[ind_verts[vpindex[vert]]];
                // set bone ids to corresponding bone_ids
                bone_ids_prop[vert] = bones[ind_verts[vpindex[vert]]];
            }
        }


        // allocate memory?
        tentacle_groups.resize(3*n_cows);
        v_ids.resize(n_cows);
        p_ids.resize(n_cows);
        cannonball_ids.resize(n_cows);

        // for 0, 1, ..., n_cows
        for (int cow_id = 0; cow_id < n_cows; ++cow_id) {

            // make octopus data
            OctopusData data;

            // cannonball displacement vector?
            Vec4 cannonball_s(0, 0, -1.5, 0.7);

            // add new particle
            // addParticle args - Vec3 &p, float r, float w, bool kinematic
            // default - kinematic = false
            cannonball_ids[cow_id] = v_scene->addParticle(
                cannonball_s.head<3>(), cannonball_s[3], 0.05);
            // idk?
            v_scene->pInfo[cannonball_ids[cow_id]].group = 1 + cow_id * 2;
            // add pill connecting itself to itself? necessary?
            v_scene->addPill(cannonball_ids[cow_id], cannonball_ids[cow_id]);
            // add (0, 0, -1.5, 0.7) to cannonball_smesh
            // cannonball has this but not pillar
            cannonball_smesh.add_sphere(
                cannonball_smesh.add_vertex(cannonball_s));

            // for each sphere
            for (int i = 0; i < spheres.size(); ++i) {
                auto &v = spheres[i];
                // w is inverse mass of sphere
                float w = 1.0 / masses[i];
                v_ids[cow_id].push_back(
                    v_scene->addParticle(v.head<3>(), v[3], w));
                // COLLISION GROUP IS SET HERE
                int add = 0;
                if (i < n_cube*n_cube*n_cube) {
                    add = 1;
                }
                v_scene->pInfo[v_ids[cow_id].back()].group = cow_id * 2 + add;

                // add radius constraints to OctopusData data
                data.radius_constraints.push_back(
                    v_scene->constraints.radius.size());
                // save radius size constraints to v_scene?
                v_scene->constraints.radius.push_back(
                    viper::C_radius(v_ids[cow_id].back(), v[3], 1e-3));

                // cannoball
                // fixes cannonballs to tentacle at ids 10 - 17?
                if (i >= 10 && i < 18) {
                    data.cannonball_constraints.push_back(
                        v_scene->constraints.distance.size());
                    v_scene->constraints.distance.push_back(viper::C_distance(
                        v_ids[cow_id].back(), cannonball_ids[cow_id],
                        cannonball_s[3], 0.f));
                }
            }

            // for each pill
            for (int i = 0; i < all_pills.size(); i++) {
                auto pill = all_pills[i];
                // add pill to v_sceen
                int p_id = v_scene->addPill(v_ids[cow_id][pill[0]],
                                            v_ids[cow_id][pill[1]]);
                // save pill id of v_scene
                p_ids[cow_id].push_back(p_id);

                // TODO - change this to activate muscle. Not sure where to change this dynamically though?
                // rest distance - defaults to euclidean distance
                float d = (spheres[pill[0]] - spheres[pill[1]]).norm();

                // save volume constraints
                data.volume_constraints.push_back(
                    v_scene->constraints.volume.size());
                v_scene->constraints.volume.push_back(
                    viper::C_volume(v_ids[cow_id][pill[0]],
                                    v_ids[cow_id][pill[1]], v_scene->state));
                // what does this compliance refer to? I thought we'd already defined it?
                float compliance = 1e-4;
                data.stretch_constraints.push_back(
                    v_scene->constraints.stretch.size());
                // to change dynamically, I think we're going to need to somehow save
                // the references to the ones we care about in v_scene.constraints.stretch
                // constraints.stretch is vector of type C_stretch
                // C_stretch - (a, b, c, L, compliance)
                // int a, b - particle ids
                // int c - pill id
                // float L - rest length
                // float compliance 
                int index = v_scene->constraints.stretch.size();
                if (original_distances.find(index) == original_distances.end()) {
                    original_distances[index] = d;
                }
                v_scene->constraints.stretch.push_back(viper::C_stretch(
                    v_ids[cow_id][pill[0]], v_ids[cow_id][pill[1]], p_id, d,
                    compliance));
                // int cube_edges = 12;
                if (index % 6 >= 3 & index < 30 * 6 - 3){
                    tentacle_groups[index % 6 - 3].push_back(index);
                }
            }
            // compares every two possibilities of pills
            // for each pill index i in 0, 1, ..., all_pills.size()
            for (int i = 0; i < all_pills.size(); ++i) {
                // for each pill index j in i+1, i+2, ..., all_pills.size()
                for (int j = i + 1; j < all_pills.size(); ++j) {
                    auto pill_i = all_pills[i];
                    auto pill_j = all_pills[j];
                    if (pill_i[0] == pill_j[0] || pill_i[0] == pill_j[1] ||
                        pill_i[1] == pill_j[0] || pill_i[1] == pill_j[1]) {
                        float compliance =
                            (compliances[i] + compliances[j]) / 2;
                        data.bend_constraints.push_back(
                            v_scene->constraints.bend.size());
                        v_scene->constraints.bend.push_back(
                            viper::C_bend(p_ids[cow_id][i], p_ids[cow_id][j],
                                          v_scene->state, compliance));
                    }
                }
            }

            // push back onto vector of OctopusData
            cow_data.push_back(data);
        }

        // TODO - where does get transforms come from?
        init_transforms = get_transforms();
        std::cout << "Uploading mesh to renderer" << std::endl;
        renderer->upload_mesh(mesh, get_transforms());
        // TODO - commenting out doesn't seem to affect?
        std::cout << "Uploading vertex texture coordinates to mesh" << std::endl;
        // WINGATE
        // The only time I've had success with loading the points for the tentacle has been when I comment out
        // the below line of code.
        // Also, it's spotty. Sometimes it displays the octopus texture on the tentacle, sometimes nothing at all
        // gpu_mesh of type GPUMesh
        renderer->get_gpu_mesh().set_vtexcoord(
            mesh.get_vertex_property<Vec2>("v:texcoord").vector());

        // defined below
        reset();

        smesh.clear();

        int offset = 0;
        // for j in 0, 1, ..., n_cows
        for (int j = 0; j < n_cows; ++j) {
            // add spheres to smesh
            for (int i = 0; i < spheres.size(); ++i) {
                auto &v = spheres[i];
                smesh.add_vertex(v);
            }

            // add control pills to smesh
            for (int i : control_pills) {
                auto pill = all_pills[i];
                smesh.add_edge(V(pill[0] + offset), V(pill[1] + offset));
            }
            // increase offset by size of spheres
            offset += spheres.size();
        }

        reset();
    }

    void reset() {
        // reset the current scene
        v_scene->reset();

        // idk
        bool pillars_active;

        // 0 - nothing
        // 1 - pillars
        // 2 - cannonballs
        // 3 - place all in same spot then explode
        switch (scene_index) {
        case 0: {
            cannonballs_active = false;
            pillars_active = false;

            // place each octopus randomly
            for (int cow_id = 0; cow_id < n_cows; ++cow_id) {
                set_position(cow_id, Vec3::Random() * 10 + Vec3(0, 12, 0),
                             Vec3::Zero());
            }
            break;
        }
        case 1: {
            cannonballs_active = false;
            pillars_active = true;

            // place each octopus randomly
            for (int cow_id = 0; cow_id < n_cows; ++cow_id) {
                set_position(cow_id, Vec3::Random() * 10 + Vec3(0, 12, 0),
                             Vec3::Zero());
            }
            break;
        }
        case 2: {
            cannonballs_active = true;
            pillars_active = false;

            // place each octopus randomly
            for (int cow_id = 0; cow_id < n_cows; ++cow_id) {
                set_position(cow_id, Vec3::Random() * 10 + Vec3(0, 12, 0),
                             Vec3::Zero());
            }
            break;
        }
        case 3: {
            cannonballs_active = false;
            pillars_active = false;

            // place each octopus at the origin, explodes
            for (int cow_id = 0; cow_id < n_cows; ++cow_id) {
                set_position(cow_id, Vec3::Zero(), Vec3::Zero());
            }
            break;
        }
        case 4: {
            cannonballs_active = false;
            pillars_active = false;

            // place each octopus randomly
            for (int cow_id = 0; cow_id < n_cows; ++cow_id) {
                set_position(cow_id, Vec3(0, 12, 0),
                             Vec3::Zero());
            }
            break;
        }
        }

        // make cannnons visible if bool
        cannonball_render_comp->visible = cannonballs_active;
        // make pillars visible if bool
        pillar_render_comp->visible = pillars_active;

        // in data, set cannonball active to current value
        for (auto &data : cow_data) {
            data.set_cannonball_enabled(cannonballs_active, *v_scene);
        }

        // for each canonball, set active or not
        for (int id : cannonball_ids) {
            v_scene->state.xa[id] = cannonballs_active;
            v_scene->state.xai[id] = cannonballs_active;

            if (!cannonballs_active) {
                v_scene->state.x[id] = Vec3(0, -2, 0);
            }
        }

        // something with pillars
        for (int i = 0; i < pillar_ids.size(); ++i) {
            float offset = pillars_active ? 0 : -5;
            // TODO - is this what fixes the positions of the pillars?
            // I don't think so... after commenting this out, doesn't change behavior
            v_scene->state.x[pillar_ids[i]][1] =
                offset + ((i % 2 == 0) ? 0 : 4);
        }
    }

    // nothing on update?
    // TODO - change constraints here? to contract muscles?
    void update(int t) {
        fix(Vec3(0.0, 0.0, 0.0));
        int delay = 240;
        float l1 = 1.4;
        float l2 = 0.2;
        if (t == delay) {
            contract(0, l1);
            contract(1, l1);
            contract(2, l2);
        } else if (t == delay*2) {
            contract(0, l1);
            contract(1, l2);
            contract(2, l1);
        } else if (t == delay*3) {
            contract(0, l2);
            contract(1, l1);
            contract(2, l1);
        } else if (t == delay*4) {
            contract(0, 1);
            contract(1, 1);
            contract(2, 1);
        }
    }

    // intersection algorithm
    int intersect(Vec3 eye, Vec3 dir) {
        float best_dist = FLT_MAX;
        int best_id = -1;

        for (auto &ids : v_ids) {
            for (int i : ids) {

                Vec3 c = v_scene->state.x[i];
                float r = v_scene->state.r[i];

                float a = dir.dot(eye - c);
                float b = -(dir.dot(eye - c));
                float det = a * a - (eye - c).squaredNorm() + r * r;

                if (det <= 0)
                    continue;

                float d0 = b + std::sqrt(det);
                float d1 = b - std::sqrt(det);

                if (d0 < 0)
                    d0 = FLT_MAX;
                if (d1 < 0)
                    d1 = FLT_MAX;
                float d = std::min(d0, d1);

                if (d < best_dist) {
                    best_dist = d;
                    best_id = i;
                }
            }
        }

        return best_id;
    }

    // TODO - when is this called?
    void vis_update() {

        // if render sphere
        if (sphere_render_comp->visible) {

            // get vertex property point
            auto vpoint = smesh.get_vertex_property<Vec4>("v:point");

            // for each cow
            for (int j = 0; j < n_cows; ++j) {
                // for each sphere
                for (int i = 0; i < spheres.size(); ++i) {
                    // idk?
                    vpoint[V(i + j * spheres.size())].head<3>() =
                        v_scene->state.x[v_ids[j][i]];
                    vpoint[V(i + j * spheres.size())][3] =
                        v_scene->state.r[v_ids[j][i]];
                }
            }

            // TODO - what is this whole new smesh thing?
            SphereMesh new_smesh;
            // add spheres at smesh vertices
            for (auto vert : smesh.vertices()) {
                auto v = new_smesh.add_vertex(vpoint[vert]);
                new_smesh.add_sphere(v);
            }

            sphere_renderer->upload_mesh(new_smesh);

            // what is this changing?
            for (auto vert : smesh.vertices()) {
                vpoint[vert][3] *= 1.02;
            }

            tsphere_renderer->upload_mesh(smesh);
        }

        // render_comp - WorldRendererComponent
        if (render_comp->visible) {

            auto transforms = get_transforms();
            renderer->upload_transforms(transforms);
        }

        // if cannonball render component
        if (cannonball_render_comp->visible) {
            auto cannonball_vpoint =
                cannonball_smesh.get_vertex_property<Vec4>("v:point");
            for (int i = 0; i < n_cows; ++i) {
                cannonball_vpoint[V(i)].head<3>() =
                    v_scene->state.x[cannonball_ids[i]];
            }

            cannonball_renderer->upload_mesh(cannonball_smesh);
        }

        if (pillar_render_comp->visible) {
            SphereMesh pillar_smesh;
            for (int i = 0; i < pillar_ids.size(); i += 2) {

                Vec3 c0 = v_scene->state.x[i];
                Vec3 c1 = v_scene->state.x[i + 1];

                float r0 = v_scene->state.r[i];
                float r1 = v_scene->state.r[i + 1];

                auto v0 = pillar_smesh.add_vertex(c0, r0);
                auto v1 = pillar_smesh.add_vertex(c1, r1);

                pillar_smesh.add_edge(v0, v1);
            }

            pillar_renderer->upload_mesh(pillar_smesh);
        }
    }

   // set position and velocity for octopus
    void set_position(int i, Vec3 pos, Vec3 v) {
        // for each sphere in octopus
        for (auto id : v_ids[i]) {
            v_scene->state.x[id] = v_scene->state.xi[id] + pos;
            v_scene->state.xp[id] = v_scene->state.x[id] - v;
            v_scene->state.r[id] = v_scene->state.ri[id];
            v_scene->state.rp[id] = v_scene->state.ri[id];
        }
        // for each pill in octopus
        for (auto id : p_ids[i]) {
            v_scene->state.q[id] = v_scene->state.qi[id];
            v_scene->state.qp[id] = v_scene->state.qi[id];
        }
        // TODO - is this what makes cannonballs move?
        // I don't think so, it also adds position to scene? but maybe
        if (cannonballs_active) {
            v_scene->state.x[cannonball_ids[i]] =
                v_scene->state.xi[cannonball_ids[i]] + pos;
            v_scene->state.xp[cannonball_ids[i]] =
                v_scene->state.x[cannonball_ids[i]] - v;
        }
    }

    void contract(int group_id, float ratio) {
        for (auto id: tentacle_groups[group_id]) {
            // std::cout << id << ": " << original_distances[id] << std::endl;
            v_scene->constraints.stretch[id].L = ratio * original_distances[id];
        }
    }


    void fix(Vec3 pos) {
        int start = n_cube * n_cube * n_cube;
        int base1 = v_ids[0][start];
        int base2 = v_ids[0][start+1];
        int base3 = v_ids[0][start+2];

        v_scene->state.x[base1] = pos;
        v_scene->state.xp[base1] = v_scene->state.x[base1];
        v_scene->state.r[base1] = v_scene->state.ri[base1];
        v_scene->state.rp[base1] = v_scene->state.ri[base1];

        Vec3 displacement = v_scene->state.xi[base1] - v_scene->state.xi[base2];
        v_scene->state.x[base2] = pos + Vec3(displacement.x(), displacement.y(), displacement.z());
        v_scene->state.xp[base2] = v_scene->state.x[base2];
        v_scene->state.r[base2] = v_scene->state.ri[base2];
        v_scene->state.rp[base2] = v_scene->state.ri[base2];

        displacement = v_scene->state.xi[base1] - v_scene->state.xi[base3];
        v_scene->state.x[base3] = pos + Vec3(displacement.x(), displacement.y(), displacement.z());
        v_scene->state.xp[base3] = v_scene->state.x[base3];
        v_scene->state.r[base3] = v_scene->state.ri[base3];
        v_scene->state.rp[base3] = v_scene->state.ri[base3];
    }

    // get_transforms, as called above
    std::vector<std::vector<Mat4x4>> get_transforms() const {

        // make transforms for each cow
        std::vector<std::vector<Mat4x4>> transforms(n_cows);

        // for each cow
        for (int j = 0; j < n_cows; ++j) {
            // for each control pill
            for (int i = 0; i < control_pills.size(); ++i) {
                int k = p_ids[j][control_pills[i]];

                int v0 = v_ids[j][all_pills[control_pills[i]][0]];
                int v1 = v_ids[j][all_pills[control_pills[i]][1]];

                // translation, rotation, scale
                Transform t0;
                t0.set_translation(v_scene->state.x[v0]);
                t0.apply_rotation(v_scene->state.q[k]);
                t0.set_scale(Vec3::Ones() * v_scene->state.r[v0]);

                transforms[j].push_back(t0.get_transformation_matrix());

                Transform t1;
                t1.set_translation(v_scene->state.x[v1]);
                t1.apply_rotation(v_scene->state.q[k]);
                t1.set_scale(Vec3::Ones() * v_scene->state.r[v1]);

                transforms[j].push_back(t1.get_transformation_matrix());
            }
        }

        return transforms;
    }
};

} // namespace OpenGP
