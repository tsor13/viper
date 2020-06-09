#pragma once

#include <cfloat>
#include <cstdlib>
// #include <ctime>
#include <fstream>
#include <iostream>
#include <thread>
#include <stdlib.h>
#include <time.h>
#include <Eigen/Dense>

// #define TINYOBJLOADER_IMPLEMENTATION
// #define USE_PNG
#include "tiny_obj_loader.h"

// VIPER PRIMITIVES
#include "Octopus.h"
// Load VIPER files
#include "Scene.h"
#include "Subprocess.h"
#include "Viper_json.h"

#include "Scene.h"
#include "CollisionGrid.cuh"

using Vec2 = Eigen::Vector2f;
using Vec3 = Eigen::Vector3f;
using Vec4 = Eigen::Vector4f;
using Vec2i = Eigen::Vector2i;
using Vec3i = Eigen::Vector3i;
using Vec4i = Eigen::Vector4i;

class Environment {
    public:
        // define scene
        viper::Scene scene;

        // // keep track of runtime performance
        // int framerate = 0;
        // double frametime = 0;
        // double sim_frametime = 0;

        // tentacle
        std::map<int,float> original_distances;
        std::vector<std::vector<int>> tentacle_groups;

        // ids
        std::vector<int> v_ids, p_ids;
        std::vector<int> cube_ids;
        std::vector<int> tentacle_ids;

        //spheres and pills
        std::vector<Vec4> spheres;
        std::vector<Vec2i> all_pills;
        std::vector<int> control_pills;
        std::vector<float> compliances;
        std::vector<float> masses;

        // cube size
        int n_cube = 3;

        // playback variables
        float playback = 1.0;
        int it_count = 10;

    void init() {
        // make an object from VIPER primitives as defined in Octopus.h
        cow::get_octopus(spheres, all_pills, control_pills, masses, compliances);

        // allocate memory
        tentacle_groups.resize(3);

        // set random seed
        srand(time(NULL));

        // populate viper scene
        // add spheres
        for (int i = 0; i < spheres.size(); i++) {
            auto &v = spheres[i];
            // TODO - what does this change?
            float w = 1.0 / masses[i];
            v_ids.push_back(
                scene.addParticle(v.head<3>(), v[3], w));

            int add;
            if (i < n_cube*n_cube*n_cube) {
                add = 0;
                cube_ids.push_back(v_ids.back());
            } else {
                add = 1;
                tentacle_ids.push_back(v_ids.back());
            }
            // COLLISION GROUP SET HERE
            scene.pInfo[v_ids.back()].group = add;

            // save radius size costraints to viper scene?
            scene.constraints.radius.push_back(
                viper::C_radius(v_ids.back(), v[3], 1e-3));

        }

        // add pills
        for (int i = 0; i < all_pills.size(); i++) {
            auto pill = all_pills[i];
            // add pill to scene
            int p_id = scene.addPill(v_ids[pill[0]], v_ids[pill[1]]);

            p_ids.push_back(p_id);

            // rest distance - defaults to euclidean distance
            float d = (spheres[pill[0]] - spheres[pill[1]]).norm();

            scene.constraints.volume.push_back(
                viper::C_volume(v_ids[pill[0]],
                                v_ids[pill[1]], scene.state));
            float compliance = 1e-4;

            int index = scene.constraints.stretch.size();
            if (original_distances.find(index) == original_distances.end()) {
                original_distances[index] = d;
            }
            scene.constraints.stretch.push_back(viper::C_stretch(
                v_ids[pill[0]], v_ids[pill[1]], p_id, d,
                compliance));
            // add to tentacle group 
            if (index % 6 < 3 && index < 30* 6 - 3) {
                tentacle_groups[index % 6].push_back(index);
            }
        }

        // add bend compliances?
        for (int i = 0; i < all_pills.size(); i++) {
            for (int j = i + 1; j < all_pills.size(); j++) {
                auto pill_i = all_pills[i];
                auto pill_j = all_pills[j];
                if (pill_i[0] == pill_j[0] || pill_i[0] == pill_j[1] ||
                    pill_i[1] == pill_j[0] || pill_i[1] == pill_j[1]) {
                    float compliance = 
                        (compliances[i] + compliances[j]) / 2;
                    scene.constraints.bend.push_back(
                        viper::C_bend(p_ids[i], p_ids[j],
                                      scene.state, compliance));
                }
            }
        }

        scene.reset();

        // rollout reset count
        int resetCount = 0;
        std::vector<float> last_action;

        // set gravity
        scene.gravity_strength = 0.1;

        // rollout actions
        int reset_delay = 1400;
        int repeat_action = 10;

        std::ofstream file;

        for (int i = 0; i < 1000 * reset_delay; i++){ 
            scene.step(playback / 60.f, it_count, true);
            // running_framerate = .9 * running_framerate + .1 / simtime;
            fix(Vec3(0.0, 0.0, 0.0));
            if (i%reset_delay == 0) {
                envReset();
                resetCount++;
            }
            if (i%repeat_action == 0) {
                auto action = randomContract();
                last_action = action;
            }
            auto s = getState();

            std::string filename;
            if (resetCount%10 < 7) {
                filename = "data/output" + std::to_string(resetCount) + ".txt";
            } else {
                filename = "data/test" + std::to_string(resetCount) + ".txt";
            }
            file.open(filename, std::ios::app);
            for (float f : s) {
                file << f << " ";
            }
            for (float f : last_action) {
                file << f << " ";
            }
            file << "\n";
            file.close();
        }
    }

    std::vector<float> getState() {
        std::vector<float> state;
        // for tentacles
        for (int ind = 0; ind < tentacle_ids.size(); ind++) {
            // get id
            int i = tentacle_ids[ind];

            Vec3 x = scene.state.x[i];
            state.push_back(x.x());
            state.push_back(x.y());
            state.push_back(x.z());
            Vec3 xp = scene.state.xp[i];
            state.push_back(xp.x());
            state.push_back(xp.y());
            state.push_back(xp.z());
            Vec3 d = x - xp;
            state.push_back(d.x());
            state.push_back(d.y());
            state.push_back(d.z());
            float r  = scene.state.r[i];
            state.push_back(r);
            float rp  = scene.state.rp[i];
            state.push_back(rp);
            float rd = r - rp;
            state.push_back(rd);
        }
        // for cube
        for (int ind = 0; ind < cube_ids.size(); ind++) {
            // get id
            int i = cube_ids[ind];

            Vec3 x = scene.state.x[i];
            state.push_back(x.x());
            state.push_back(x.y());
            state.push_back(x.z());
            Vec3 xp = scene.state.xp[i];
            state.push_back(xp.x());
            state.push_back(xp.y());
            state.push_back(xp.z());
            Vec3 d = x - xp;
            state.push_back(d.x());
            state.push_back(d.y());
            state.push_back(d.z());
            float r  = scene.state.r[i];
            state.push_back(r);
            float rp  = scene.state.rp[i];
            state.push_back(rp);
            float rd = r - rp;
            state.push_back(rd);
        }
        return state;
    }

    void contract(int group_id, float ratio) {
        for (auto id: tentacle_groups[group_id]) {
            scene.constraints.stretch[id].L = ratio * original_distances[id];
        }
    }

    std::vector<float> randomContract(float min = .1, float max = 2) {
        std::vector<float> action;
        float r1 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        float r2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        float r3 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        // map to desired range
        r1 = r1 * min + (1-r1) * max;
        r2 = r2 * min + (1-r2) * max;
        r3 = r3 * min + (1-r3) * max;
        // normalize
        r1 *= 3 / (r1 + r2 + r3);
        r2 *= 3 / (r1 + r2 + r3);
        r3 *= 3 / (r1 + r2 + r3);
        // contract and save action
        contract(0, r1);
        action.push_back(r1);
        contract(1, r2);
        action.push_back(r2);
        contract(2, r3);
        action.push_back(r3);
        return action;
    }

    void envReset() {
        int base = v_ids[n_cube * n_cube * n_cube];
        // reset to initial position around pos
        Vec3 pos = Vec3(0, 0, 0);
        // random direction for initial velocity
        Vec3 random_vector = Vec3::Random() * 0.1;
        // cube displacement
        Vec3 cube_displacement = Vec3::Random() * 10;
        // for each sphere
        for (int ind = 0; ind < tentacle_ids.size(); ind++) {
            // get id
            int i = tentacle_ids[ind];
            // set position to relative position where base is at pos
            scene.state.x[i] = pos - scene.state.xi[base] + scene.state.xi[i];
            // change the previous slightly to induce a random initial velocity
            scene.state.xp[i] = scene.state.x[i] + random_vector;
            // set rotation to initial
            scene.state.r[i] = scene.state.ri[i];
            scene.state.rp[i] = scene.state.ri[i];
        }
        for (int ind = 0; ind < cube_ids.size(); ind++) {
            // get id
            int i = cube_ids[ind];
            // set position to relative position where base is at pos
            scene.state.x[i] = pos - scene.state.xi[base] + scene.state.xi[i] + cube_displacement;
            // change the previous slightly to induce a random initial velocity
            scene.state.xp[i] = scene.state.x[i] + random_vector;
            // set rotation to initial
            scene.state.r[i] = scene.state.ri[i];
            scene.state.rp[i] = scene.state.ri[i];
        }
    }

    void fix(Vec3 pos) {
        int start = n_cube * n_cube * n_cube;
        int base1 = v_ids[start];
        int base2 = v_ids[start+1];
        int base3 = v_ids[start+2];

        scene.state.x[base1] = pos;
        scene.state.xp[base1] = scene.state.x[base1];
        scene.state.r[base1] = scene.state.ri[base1];
        scene.state.rp[base1] = scene.state.ri[base1];

        scene.state.x[base2] = pos - scene.state.xi[base1] + scene.state.xi[base2];
        scene.state.xp[base2] = scene.state.x[base2];
        scene.state.r[base2] = scene.state.ri[base2];
        scene.state.rp[base2] = scene.state.ri[base2];

        scene.state.x[base3] = pos - scene.state.xi[base1] + scene.state.xi[base3];
        scene.state.xp[base3] = scene.state.x[base3];
        scene.state.r[base3] = scene.state.ri[base3];
        scene.state.rp[base3] = scene.state.ri[base3];
    }

    std::vector<float> tick(float r1, float r2, float r3) {
        contract(0, r1);
        contract(1, r2);
        contract(2, r3);
        scene.step(playback / 60.f, it_count, true);
        return getState();
    }
};
