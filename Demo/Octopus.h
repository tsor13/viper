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

#pragma once

// include packages
#include <map>
#include <vector>

#include <Eigen/Dense>

// make cow namespace
namespace cow {

// define vectors
// TODO - what represent?
using Vec2 = Eigen::Vector2f;
using Vec3 = Eigen::Vector3f;
using Vec4 = Eigen::Vector4f;

using Vec2i = Eigen::Vector2i;
using Vec3i = Eigen::Vector3i;
using Vec4i = Eigen::Vector4i;

// make an octopus given spheres, pills, etc
inline void get_octopus(std::vector<Vec4> &spheres, std::vector<Vec2i> &pills,
                        std::vector<int> &control_pills,
                        std::vector<float> &masses,
                        std::vector<float> &compliances) {

    // TODO - what is clearing?
    spheres.clear();
    pills.clear();
    control_pills.clear();
    masses.clear();
    compliances.clear();

    // scale
    //auto scale = 0.01f

    // Define sphere vals
    // (x, y, z, r)
    std::map<std::string, Vec4> sphere_vals = {
        {"tentacle1.1", 0.500*Vec4(-2.450, -23.800, 1.280, 1.200)},
        {"tentacle1.2", 0.500*Vec4(-2.450, -23.800, -1.150, 1.200)},
        {"tentacle1.3", 0.500*Vec4(-4.530, -23.800, 0.130, 1.200)},
        {"tentacle2.1", 0.500*Vec4(-2.374, -22.269, 1.270, 1.190)},
        {"tentacle2.2", 0.500*Vec4(-2.374, -22.269, -0.909, 1.190)},
        {"tentacle2.3", 0.500*Vec4(-4.436, -22.269, 0.133, 1.190)},
        {"tentacle3.1", 0.500*Vec4(-2.299, -20.738, 1.259, 1.179)},
        {"tentacle3.2", 0.500*Vec4(-2.299, -20.738, -0.668, 1.179)},
        {"tentacle3.3", 0.500*Vec4(-4.342, -20.738, 0.136, 1.179)},
        {"tentacle4.1", 0.500*Vec4(-2.223, -19.207, 1.249, 1.169)},
        {"tentacle4.2", 0.500*Vec4(-2.223, -19.207, -0.427, 1.169)},
        {"tentacle4.3", 0.500*Vec4(-4.248, -19.207, 0.139, 1.169)},
        {"tentacle5.1", 0.500*Vec4(-2.148, -17.676, 1.239, 1.159)},
        {"tentacle5.2", 0.500*Vec4(-2.148, -17.676, -0.186, 1.159)},
        {"tentacle5.3", 0.500*Vec4(-4.153, -17.676, 0.142, 1.159)},
        {"tentacle6.1", 0.500*Vec4(-2.072, -16.145, 1.228, 1.148)},
        {"tentacle6.2", 0.500*Vec4(-2.072, -16.145, 0.055, 1.148)},
        {"tentacle6.3", 0.500*Vec4(-4.059, -16.145, 0.146, 1.148)},
        {"tentacle7.1", 0.500*Vec4(-1.997, -14.614, 1.218, 1.138)},
        {"tentacle7.2", 0.500*Vec4(-1.997, -14.614, 0.296, 1.138)},
        {"tentacle7.3", 0.500*Vec4(-3.965, -14.614, 0.149, 1.138)},
        {"tentacle8.1", 0.500*Vec4(-1.921, -13.083, 1.208, 1.128)},
        {"tentacle8.2", 0.500*Vec4(-1.921, -13.083, 0.537, 1.128)},
        {"tentacle8.3", 0.500*Vec4(-3.871, -13.083, 0.152, 1.128)},
        {"tentacle9.1", 0.500*Vec4(-1.846, -11.552, 1.197, 1.117)},
        {"tentacle9.2", 0.500*Vec4(-1.846, -11.552, 0.778, 1.117)},
        {"tentacle9.3", 0.500*Vec4(-3.777, -11.552, 0.155, 1.117)},
        {"tentacle10.1", 0.500*Vec4(-1.770, -10.021, 1.187, 1.107)},
        {"tentacle10.2", 0.500*Vec4(-1.770, -10.021, 1.019, 1.107)},
        {"tentacle10.3", 0.500*Vec4(-3.683, -10.021, 0.158, 1.107)},
        {"tentacle11.1", 0.500*Vec4(-1.695, -8.414, 1.162, 1.088)},
        {"tentacle11.2", 0.500*Vec4(-1.695, -8.414, 1.109, 1.088)},
        {"tentacle11.3", 0.500*Vec4(-3.577, -8.414, 0.151, 1.088)},
        {"tentacle12.1", 0.500*Vec4(-1.621, -6.655, 1.108, 1.052)},
        {"tentacle12.2", 0.500*Vec4(-1.621, -6.655, 0.896, 1.052)},
        {"tentacle12.3", 0.500*Vec4(-3.449, -6.655, 0.124, 1.052)},
        {"tentacle13.1", 0.500*Vec4(-1.546, -4.897, 1.054, 1.016)},
        {"tentacle13.2", 0.500*Vec4(-1.546, -4.897, 0.683, 1.016)},
        {"tentacle13.3", 0.500*Vec4(-3.321, -4.897, 0.097, 1.016)},
        {"tentacle14.1", 0.500*Vec4(-1.472, -3.138, 1.001, 0.979)},
        {"tentacle14.2", 0.500*Vec4(-1.472, -3.138, 0.470, 0.979)},
        {"tentacle14.3", 0.500*Vec4(-3.192, -3.138, 0.070, 0.979)},
        {"tentacle15.1", 0.500*Vec4(-1.397, -1.379, 0.947, 0.943)},
        {"tentacle15.2", 0.500*Vec4(-1.397, -1.379, 0.257, 0.943)},
        {"tentacle15.3", 0.500*Vec4(-3.064, -1.379, 0.043, 0.943)},
        {"tentacle16.1", 0.500*Vec4(-1.323, 0.379, 0.893, 0.907)},
        {"tentacle16.2", 0.500*Vec4(-1.323, 0.379, 0.043, 0.907)},
        {"tentacle16.3", 0.500*Vec4(-2.936, 0.379, 0.017, 0.907)},
        {"tentacle17.1", 0.500*Vec4(-1.248, 2.138, 0.839, 0.871)},
        {"tentacle17.2", 0.500*Vec4(-1.248, 2.138, -0.170, 0.871)},
        {"tentacle17.3", 0.500*Vec4(-2.808, 2.138, -0.010, 0.871)},
        {"tentacle18.1", 0.500*Vec4(-1.174, 3.897, 0.786, 0.834)},
        {"tentacle18.2", 0.500*Vec4(-1.174, 3.897, -0.383, 0.834)},
        {"tentacle18.3", 0.500*Vec4(-2.679, 3.897, -0.037, 0.834)},
        {"tentacle19.1", 0.500*Vec4(-1.099, 5.655, 0.732, 0.798)},
        {"tentacle19.2", 0.500*Vec4(-1.099, 5.655, -0.596, 0.798)},
        {"tentacle19.3", 0.500*Vec4(-2.551, 5.655, -0.064, 0.798)},
        {"tentacle20.1", 0.500*Vec4(-1.025, 7.414, 0.678, 0.762)},
        {"tentacle20.2", 0.500*Vec4(-1.025, 7.414, -0.809, 0.762)},
        {"tentacle20.3", 0.500*Vec4(-2.423, 7.414, -0.091, 0.762)},
        {"tentacle21.1", 0.500*Vec4(-0.970, 9.103, 0.609, 0.712)},
        {"tentacle21.2", 0.500*Vec4(-0.970, 9.103, -0.841, 0.712)},
        {"tentacle21.3", 0.500*Vec4(-2.277, 9.103, -0.101, 0.712)},
        {"tentacle22.1", 0.500*Vec4(-0.924, 10.759, 0.532, 0.655)},
        {"tentacle22.2", 0.500*Vec4(-0.924, 10.759, -0.782, 0.655)},
        {"tentacle22.3", 0.500*Vec4(-2.123, 10.759, -0.103, 0.655)},
        {"tentacle23.1", 0.500*Vec4(-0.879, 12.414, 0.456, 0.598)},
        {"tentacle23.2", 0.500*Vec4(-0.879, 12.414, -0.723, 0.598)},
        {"tentacle23.3", 0.500*Vec4(-1.969, 12.414, -0.106, 0.598)},
        {"tentacle24.1", 0.500*Vec4(-0.833, 14.069, 0.379, 0.541)},
        {"tentacle24.2", 0.500*Vec4(-0.833, 14.069, -0.664, 0.541)},
        {"tentacle24.3", 0.500*Vec4(-1.815, 14.069, -0.108, 0.541)},
        {"tentacle25.1", 0.500*Vec4(-0.788, 15.724, 0.303, 0.484)},
        {"tentacle25.2", 0.500*Vec4(-0.788, 15.724, -0.605, 0.484)},
        {"tentacle25.3", 0.500*Vec4(-1.661, 15.724, -0.110, 0.484)},
        {"tentacle26.1", 0.500*Vec4(-0.742, 17.379, 0.226, 0.428)},
        {"tentacle26.2", 0.500*Vec4(-0.742, 17.379, -0.546, 0.428)},
        {"tentacle26.3", 0.500*Vec4(-1.507, 17.379, -0.112, 0.428)},
        {"tentacle27.1", 0.500*Vec4(-0.697, 19.034, 0.150, 0.371)},
        {"tentacle27.2", 0.500*Vec4(-0.697, 19.034, -0.487, 0.371)},
        {"tentacle27.3", 0.500*Vec4(-1.352, 19.034, -0.114, 0.371)},
        {"tentacle28.1", 0.500*Vec4(-0.651, 20.690, 0.073, 0.314)},
        {"tentacle28.2", 0.500*Vec4(-0.651, 20.690, -0.428, 0.314)},
        {"tentacle28.3", 0.500*Vec4(-1.198, 20.690, -0.116, 0.314)},
        {"tentacle29.1", 0.500*Vec4(-0.606, 22.345, -0.003, 0.257)},
        {"tentacle29.2", 0.500*Vec4(-0.606, 22.345, -0.369, 0.257)},
        {"tentacle29.3", 0.500*Vec4(-1.044, 22.345, -0.118, 0.257)},
        {"tentacle30.1", 0.500*Vec4(-0.560, 24.000, -0.080, 0.200)},
        {"tentacle30.2", 0.500*Vec4(-0.560, 24.000, -0.310, 0.200)},
        {"tentacle30.3", 0.500*Vec4(-0.890, 24.000, -0.120, 0.200)},

        {"cube1.1.1", 0.500*Vec4(-1.333, -1.333, -1.333, 0.667)},
        {"cube1.1.2", 0.500*Vec4(-1.333, -1.333, 0.000, 0.667)},
        {"cube1.1.3", 0.500*Vec4(-1.333, -1.333, 1.333, 0.667)},
        {"cube1.2.1", 0.500*Vec4(-1.333, 0.000, -1.333, 0.667)},
        {"cube1.2.2", 0.500*Vec4(-1.333, 0.000, 0.000, 0.667)},
        {"cube1.2.3", 0.500*Vec4(-1.333, 0.000, 1.333, 0.667)},
        {"cube1.3.1", 0.500*Vec4(-1.333, 1.333, -1.333, 0.667)},
        {"cube1.3.2", 0.500*Vec4(-1.333, 1.333, 0.000, 0.667)},
        {"cube1.3.3", 0.500*Vec4(-1.333, 1.333, 1.333, 0.667)},
        {"cube2.1.1", 0.500*Vec4(0.000, -1.333, -1.333, 0.667)},
        {"cube2.1.2", 0.500*Vec4(0.000, -1.333, 0.000, 0.667)},
        {"cube2.1.3", 0.500*Vec4(0.000, -1.333, 1.333, 0.667)},
        {"cube2.2.1", 0.500*Vec4(0.000, 0.000, -1.333, 0.667)},
        {"cube2.2.2", 0.500*Vec4(0.000, 0.000, 0.000, 0.667)},
        {"cube2.2.3", 0.500*Vec4(0.000, 0.000, 1.333, 0.667)},
        {"cube2.3.1", 0.500*Vec4(0.000, 1.333, -1.333, 0.667)},
        {"cube2.3.2", 0.500*Vec4(0.000, 1.333, 0.000, 0.667)},
        {"cube2.3.3", 0.500*Vec4(0.000, 1.333, 1.333, 0.667)},
        {"cube3.1.1", 0.500*Vec4(1.333, -1.333, -1.333, 0.667)},
        {"cube3.1.2", 0.500*Vec4(1.333, -1.333, 0.000, 0.667)},
        {"cube3.1.3", 0.500*Vec4(1.333, -1.333, 1.333, 0.667)},
        {"cube3.2.1", 0.500*Vec4(1.333, 0.000, -1.333, 0.667)},
        {"cube3.2.2", 0.500*Vec4(1.333, 0.000, 0.000, 0.667)},
        {"cube3.2.3", 0.500*Vec4(1.333, 0.000, 1.333, 0.667)},
        {"cube3.3.1", 0.500*Vec4(1.333, 1.333, -1.333, 0.667)},
        {"cube3.3.2", 0.500*Vec4(1.333, 1.333, 0.000, 0.667)},
        {"cube3.3.3", 0.500*Vec4(1.333, 1.333, 1.333, 0.667)},
    };

    std::map<std::string, float> sphere_masses = {
        {"tentacle1.1", 0.360f},
        {"tentacle1.2", 0.360f},
        {"tentacle1.3", 0.360f},
        {"tentacle2.1", 0.354f},
        {"tentacle2.2", 0.354f},
        {"tentacle2.3", 0.354f},
        {"tentacle3.1", 0.348f},
        {"tentacle3.2", 0.348f},
        {"tentacle3.3", 0.348f},
        {"tentacle4.1", 0.342f},
        {"tentacle4.2", 0.342f},
        {"tentacle4.3", 0.342f},
        {"tentacle5.1", 0.336f},
        {"tentacle5.2", 0.336f},
        {"tentacle5.3", 0.336f},
        {"tentacle6.1", 0.330f},
        {"tentacle6.2", 0.330f},
        {"tentacle6.3", 0.330f},
        {"tentacle7.1", 0.324f},
        {"tentacle7.2", 0.324f},
        {"tentacle7.3", 0.324f},
        {"tentacle8.1", 0.318f},
        {"tentacle8.2", 0.318f},
        {"tentacle8.3", 0.318f},
        {"tentacle9.1", 0.312f},
        {"tentacle9.2", 0.312f},
        {"tentacle9.3", 0.312f},
        {"tentacle10.1", 0.306f},
        {"tentacle10.2", 0.306f},
        {"tentacle10.3", 0.306f},
        {"tentacle11.1", 0.296f},
        {"tentacle11.2", 0.296f},
        {"tentacle11.3", 0.296f},
        {"tentacle12.1", 0.277f},
        {"tentacle12.2", 0.277f},
        {"tentacle12.3", 0.277f},
        {"tentacle13.1", 0.258f},
        {"tentacle13.2", 0.258f},
        {"tentacle13.3", 0.258f},
        {"tentacle14.1", 0.240f},
        {"tentacle14.2", 0.240f},
        {"tentacle14.3", 0.240f},
        {"tentacle15.1", 0.222f},
        {"tentacle15.2", 0.222f},
        {"tentacle15.3", 0.222f},
        {"tentacle16.1", 0.206f},
        {"tentacle16.2", 0.206f},
        {"tentacle16.3", 0.206f},
        {"tentacle17.1", 0.190f},
        {"tentacle17.2", 0.190f},
        {"tentacle17.3", 0.190f},
        {"tentacle18.1", 0.174f},
        {"tentacle18.2", 0.174f},
        {"tentacle18.3", 0.174f},
        {"tentacle19.1", 0.159f},
        {"tentacle19.2", 0.159f},
        {"tentacle19.3", 0.159f},
        {"tentacle20.1", 0.145f},
        {"tentacle20.2", 0.145f},
        {"tentacle20.3", 0.145f},
        {"tentacle21.1", 0.127f},
        {"tentacle21.2", 0.127f},
        {"tentacle21.3", 0.127f},
        {"tentacle22.1", 0.107f},
        {"tentacle22.2", 0.107f},
        {"tentacle22.3", 0.107f},
        {"tentacle23.1", 0.089f},
        {"tentacle23.2", 0.089f},
        {"tentacle23.3", 0.089f},
        {"tentacle24.1", 0.073f},
        {"tentacle24.2", 0.073f},
        {"tentacle24.3", 0.073f},
        {"tentacle25.1", 0.059f},
        {"tentacle25.2", 0.059f},
        {"tentacle25.3", 0.059f},
        {"tentacle26.1", 0.046f},
        {"tentacle26.2", 0.046f},
        {"tentacle26.3", 0.046f},
        {"tentacle27.1", 0.034f},
        {"tentacle27.2", 0.034f},
        {"tentacle27.3", 0.034f},
        {"tentacle28.1", 0.025f},
        {"tentacle28.2", 0.025f},
        {"tentacle28.3", 0.025f},
        {"tentacle29.1", 0.016f},
        {"tentacle29.2", 0.016f},
        {"tentacle29.3", 0.016f},
        {"tentacle30.1", 0.010f},
        {"tentacle30.2", 0.010f},
        {"tentacle30.3", 0.010f},

        {"cube1.1.1", 0.037f},
        {"cube1.1.2", 0.037f},
        {"cube1.1.3", 0.037f},
        {"cube1.2.1", 0.037f},
        {"cube1.2.2", 0.037f},
        {"cube1.2.3", 0.037f},
        {"cube1.3.1", 0.037f},
        {"cube1.3.2", 0.037f},
        {"cube1.3.3", 0.037f},
        {"cube2.1.1", 0.037f},
        {"cube2.1.2", 0.037f},
        {"cube2.1.3", 0.037f},
        {"cube2.2.1", 0.037f},
        {"cube2.2.2", 0.037f},
        {"cube2.2.3", 0.037f},
        {"cube2.3.1", 0.037f},
        {"cube2.3.2", 0.037f},
        {"cube2.3.3", 0.037f},
        {"cube3.1.1", 0.037f},
        {"cube3.1.2", 0.037f},
        {"cube3.1.3", 0.037f},
        {"cube3.2.1", 0.037f},
        {"cube3.2.2", 0.037f},
        {"cube3.2.3", 0.037f},
        {"cube3.3.1", 0.037f},
        {"cube3.3.2", 0.037f},
        {"cube3.3.3", 0.037f},
    };
    // make ids for each sphere
    std::map<std::string, int> sphere_ids;
    int i = 0;
    for (auto pair : sphere_vals) {
        sphere_ids[pair.first] = i++;
        spheres.push_back(pair.second);
        masses.push_back(sphere_masses[pair.first]);
    }

    // make tuples for each pill connection?
    using tuple = std::tuple<Vec2i, bool, float>;
    // last element in tuple is compliance, lower is stiffer, higher is more compliant
    std::vector<tuple> pill_flags = {
        tuple(Vec2i(sphere_ids["tentacle1.1"], sphere_ids["tentacle2.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle1.2"], sphere_ids["tentacle2.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle1.3"], sphere_ids["tentacle2.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle2.1"], sphere_ids["tentacle3.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle2.2"], sphere_ids["tentacle3.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle2.3"], sphere_ids["tentacle3.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle3.1"], sphere_ids["tentacle4.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle3.2"], sphere_ids["tentacle4.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle3.3"], sphere_ids["tentacle4.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle4.1"], sphere_ids["tentacle5.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle4.2"], sphere_ids["tentacle5.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle4.3"], sphere_ids["tentacle5.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle5.1"], sphere_ids["tentacle6.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle5.2"], sphere_ids["tentacle6.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle5.3"], sphere_ids["tentacle6.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle6.1"], sphere_ids["tentacle7.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle6.2"], sphere_ids["tentacle7.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle6.3"], sphere_ids["tentacle7.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle7.1"], sphere_ids["tentacle8.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle7.2"], sphere_ids["tentacle8.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle7.3"], sphere_ids["tentacle8.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle8.1"], sphere_ids["tentacle9.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle8.2"], sphere_ids["tentacle9.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle8.3"], sphere_ids["tentacle9.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle9.1"], sphere_ids["tentacle10.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle9.2"], sphere_ids["tentacle10.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle9.3"], sphere_ids["tentacle10.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle10.1"], sphere_ids["tentacle11.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle10.2"], sphere_ids["tentacle11.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle10.3"], sphere_ids["tentacle11.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle11.1"], sphere_ids["tentacle12.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle11.2"], sphere_ids["tentacle12.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle11.3"], sphere_ids["tentacle12.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle12.1"], sphere_ids["tentacle13.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle12.2"], sphere_ids["tentacle13.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle12.3"], sphere_ids["tentacle13.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle13.1"], sphere_ids["tentacle14.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle13.2"], sphere_ids["tentacle14.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle13.3"], sphere_ids["tentacle14.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle14.1"], sphere_ids["tentacle15.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle14.2"], sphere_ids["tentacle15.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle14.3"], sphere_ids["tentacle15.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle15.1"], sphere_ids["tentacle16.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle15.2"], sphere_ids["tentacle16.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle15.3"], sphere_ids["tentacle16.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle16.1"], sphere_ids["tentacle17.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle16.2"], sphere_ids["tentacle17.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle16.3"], sphere_ids["tentacle17.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle17.1"], sphere_ids["tentacle18.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle17.2"], sphere_ids["tentacle18.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle17.3"], sphere_ids["tentacle18.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle18.1"], sphere_ids["tentacle19.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle18.2"], sphere_ids["tentacle19.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle18.3"], sphere_ids["tentacle19.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle19.1"], sphere_ids["tentacle20.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle19.2"], sphere_ids["tentacle20.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle19.3"], sphere_ids["tentacle20.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle20.1"], sphere_ids["tentacle21.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle20.2"], sphere_ids["tentacle21.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle20.3"], sphere_ids["tentacle21.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle21.1"], sphere_ids["tentacle22.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle21.2"], sphere_ids["tentacle22.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle21.3"], sphere_ids["tentacle22.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle22.1"], sphere_ids["tentacle23.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle22.2"], sphere_ids["tentacle23.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle22.3"], sphere_ids["tentacle23.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle23.1"], sphere_ids["tentacle24.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle23.2"], sphere_ids["tentacle24.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle23.3"], sphere_ids["tentacle24.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle24.1"], sphere_ids["tentacle25.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle24.2"], sphere_ids["tentacle25.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle24.3"], sphere_ids["tentacle25.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle25.1"], sphere_ids["tentacle26.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle25.2"], sphere_ids["tentacle26.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle25.3"], sphere_ids["tentacle26.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle26.1"], sphere_ids["tentacle27.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle26.2"], sphere_ids["tentacle27.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle26.3"], sphere_ids["tentacle27.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle27.1"], sphere_ids["tentacle28.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle27.2"], sphere_ids["tentacle28.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle27.3"], sphere_ids["tentacle28.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle28.1"], sphere_ids["tentacle29.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle28.2"], sphere_ids["tentacle29.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle28.3"], sphere_ids["tentacle29.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle29.1"], sphere_ids["tentacle30.1"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle29.2"], sphere_ids["tentacle30.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle29.3"], sphere_ids["tentacle30.3"]), true, 1e-07f),



        tuple(Vec2i(sphere_ids["tentacle1.1"], sphere_ids["tentacle1.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle1.1"], sphere_ids["tentacle1.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle1.2"], sphere_ids["tentacle1.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle2.1"], sphere_ids["tentacle2.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle2.1"], sphere_ids["tentacle2.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle2.2"], sphere_ids["tentacle2.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle3.1"], sphere_ids["tentacle3.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle3.1"], sphere_ids["tentacle3.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle3.2"], sphere_ids["tentacle3.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle4.1"], sphere_ids["tentacle4.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle4.1"], sphere_ids["tentacle4.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle4.2"], sphere_ids["tentacle4.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle5.1"], sphere_ids["tentacle5.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle5.1"], sphere_ids["tentacle5.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle5.2"], sphere_ids["tentacle5.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle6.1"], sphere_ids["tentacle6.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle6.1"], sphere_ids["tentacle6.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle6.2"], sphere_ids["tentacle6.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle7.1"], sphere_ids["tentacle7.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle7.1"], sphere_ids["tentacle7.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle7.2"], sphere_ids["tentacle7.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle8.1"], sphere_ids["tentacle8.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle8.1"], sphere_ids["tentacle8.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle8.2"], sphere_ids["tentacle8.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle9.1"], sphere_ids["tentacle9.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle9.1"], sphere_ids["tentacle9.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle9.2"], sphere_ids["tentacle9.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle10.1"], sphere_ids["tentacle10.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle10.1"], sphere_ids["tentacle10.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle10.2"], sphere_ids["tentacle10.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle11.1"], sphere_ids["tentacle11.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle11.1"], sphere_ids["tentacle11.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle11.2"], sphere_ids["tentacle11.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle12.1"], sphere_ids["tentacle12.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle12.1"], sphere_ids["tentacle12.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle12.2"], sphere_ids["tentacle12.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle13.1"], sphere_ids["tentacle13.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle13.1"], sphere_ids["tentacle13.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle13.2"], sphere_ids["tentacle13.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle14.1"], sphere_ids["tentacle14.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle14.1"], sphere_ids["tentacle14.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle14.2"], sphere_ids["tentacle14.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle15.1"], sphere_ids["tentacle15.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle15.1"], sphere_ids["tentacle15.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle15.2"], sphere_ids["tentacle15.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle16.1"], sphere_ids["tentacle16.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle16.1"], sphere_ids["tentacle16.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle16.2"], sphere_ids["tentacle16.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle17.1"], sphere_ids["tentacle17.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle17.1"], sphere_ids["tentacle17.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle17.2"], sphere_ids["tentacle17.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle18.1"], sphere_ids["tentacle18.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle18.1"], sphere_ids["tentacle18.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle18.2"], sphere_ids["tentacle18.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle19.1"], sphere_ids["tentacle19.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle19.1"], sphere_ids["tentacle19.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle19.2"], sphere_ids["tentacle19.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle20.1"], sphere_ids["tentacle20.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle20.1"], sphere_ids["tentacle20.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle20.2"], sphere_ids["tentacle20.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle21.1"], sphere_ids["tentacle21.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle21.1"], sphere_ids["tentacle21.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle21.2"], sphere_ids["tentacle21.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle22.1"], sphere_ids["tentacle22.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle22.1"], sphere_ids["tentacle22.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle22.2"], sphere_ids["tentacle22.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle23.1"], sphere_ids["tentacle23.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle23.1"], sphere_ids["tentacle23.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle23.2"], sphere_ids["tentacle23.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle24.1"], sphere_ids["tentacle24.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle24.1"], sphere_ids["tentacle24.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle24.2"], sphere_ids["tentacle24.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle25.1"], sphere_ids["tentacle25.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle25.1"], sphere_ids["tentacle25.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle25.2"], sphere_ids["tentacle25.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle26.1"], sphere_ids["tentacle26.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle26.1"], sphere_ids["tentacle26.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle26.2"], sphere_ids["tentacle26.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle27.1"], sphere_ids["tentacle27.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle27.1"], sphere_ids["tentacle27.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle27.2"], sphere_ids["tentacle27.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle28.1"], sphere_ids["tentacle28.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle28.1"], sphere_ids["tentacle28.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle28.2"], sphere_ids["tentacle28.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle29.1"], sphere_ids["tentacle29.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle29.1"], sphere_ids["tentacle29.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle29.2"], sphere_ids["tentacle29.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["tentacle30.1"], sphere_ids["tentacle30.2"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle30.1"], sphere_ids["tentacle30.3"]), true, 1e-07f),
        tuple(Vec2i(sphere_ids["tentacle30.2"], sphere_ids["tentacle30.3"]), true, 1e-07f),

        tuple(Vec2i(sphere_ids["cube1.1.1"], sphere_ids["cube2.1.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.1.1"], sphere_ids["cube1.2.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.1.1"], sphere_ids["cube1.1.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.1.2"], sphere_ids["cube2.1.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.1.2"], sphere_ids["cube1.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.1.2"], sphere_ids["cube1.1.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.1.2"], sphere_ids["cube1.1.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.1.3"], sphere_ids["cube2.1.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.1.3"], sphere_ids["cube1.2.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.1.3"], sphere_ids["cube1.1.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.2.1"], sphere_ids["cube2.2.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.2.1"], sphere_ids["cube1.1.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.2.1"], sphere_ids["cube1.3.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.2.1"], sphere_ids["cube1.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.2.2"], sphere_ids["cube2.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.2.2"], sphere_ids["cube1.1.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.2.2"], sphere_ids["cube1.3.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.2.2"], sphere_ids["cube1.2.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.2.2"], sphere_ids["cube1.2.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.2.3"], sphere_ids["cube2.2.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.2.3"], sphere_ids["cube1.1.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.2.3"], sphere_ids["cube1.3.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.2.3"], sphere_ids["cube1.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.3.1"], sphere_ids["cube2.3.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.3.1"], sphere_ids["cube1.2.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.3.1"], sphere_ids["cube1.3.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.3.2"], sphere_ids["cube2.3.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.3.2"], sphere_ids["cube1.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.3.2"], sphere_ids["cube1.3.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.3.2"], sphere_ids["cube1.3.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.3.3"], sphere_ids["cube2.3.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.3.3"], sphere_ids["cube1.2.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube1.3.3"], sphere_ids["cube1.3.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.1.1"], sphere_ids["cube1.1.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.1.1"], sphere_ids["cube3.1.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.1.1"], sphere_ids["cube2.2.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.1.1"], sphere_ids["cube2.1.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.1.2"], sphere_ids["cube1.1.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.1.2"], sphere_ids["cube3.1.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.1.2"], sphere_ids["cube2.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.1.2"], sphere_ids["cube2.1.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.1.2"], sphere_ids["cube2.1.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.1.3"], sphere_ids["cube1.1.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.1.3"], sphere_ids["cube3.1.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.1.3"], sphere_ids["cube2.2.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.1.3"], sphere_ids["cube2.1.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.1"], sphere_ids["cube1.2.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.1"], sphere_ids["cube3.2.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.1"], sphere_ids["cube2.1.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.1"], sphere_ids["cube2.3.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.1"], sphere_ids["cube2.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.2"], sphere_ids["cube1.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.2"], sphere_ids["cube3.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.2"], sphere_ids["cube2.1.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.2"], sphere_ids["cube2.3.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.2"], sphere_ids["cube2.2.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.2"], sphere_ids["cube2.2.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.3"], sphere_ids["cube1.2.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.3"], sphere_ids["cube3.2.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.3"], sphere_ids["cube2.1.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.3"], sphere_ids["cube2.3.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.2.3"], sphere_ids["cube2.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.3.1"], sphere_ids["cube1.3.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.3.1"], sphere_ids["cube3.3.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.3.1"], sphere_ids["cube2.2.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.3.1"], sphere_ids["cube2.3.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.3.2"], sphere_ids["cube1.3.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.3.2"], sphere_ids["cube3.3.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.3.2"], sphere_ids["cube2.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.3.2"], sphere_ids["cube2.3.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.3.2"], sphere_ids["cube2.3.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.3.3"], sphere_ids["cube1.3.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.3.3"], sphere_ids["cube3.3.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.3.3"], sphere_ids["cube2.2.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube2.3.3"], sphere_ids["cube2.3.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.1.1"], sphere_ids["cube2.1.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.1.1"], sphere_ids["cube3.2.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.1.1"], sphere_ids["cube3.1.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.1.2"], sphere_ids["cube2.1.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.1.2"], sphere_ids["cube3.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.1.2"], sphere_ids["cube3.1.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.1.2"], sphere_ids["cube3.1.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.1.3"], sphere_ids["cube2.1.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.1.3"], sphere_ids["cube3.2.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.1.3"], sphere_ids["cube3.1.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.2.1"], sphere_ids["cube2.2.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.2.1"], sphere_ids["cube3.1.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.2.1"], sphere_ids["cube3.3.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.2.1"], sphere_ids["cube3.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.2.2"], sphere_ids["cube2.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.2.2"], sphere_ids["cube3.1.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.2.2"], sphere_ids["cube3.3.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.2.2"], sphere_ids["cube3.2.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.2.2"], sphere_ids["cube3.2.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.2.3"], sphere_ids["cube2.2.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.2.3"], sphere_ids["cube3.1.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.2.3"], sphere_ids["cube3.3.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.2.3"], sphere_ids["cube3.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.3.1"], sphere_ids["cube2.3.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.3.1"], sphere_ids["cube3.2.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.3.1"], sphere_ids["cube3.3.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.3.2"], sphere_ids["cube2.3.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.3.2"], sphere_ids["cube3.2.2"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.3.2"], sphere_ids["cube3.3.1"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.3.2"], sphere_ids["cube3.3.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.3.3"], sphere_ids["cube2.3.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.3.3"], sphere_ids["cube3.2.3"]), false, 0.0f),
        tuple(Vec2i(sphere_ids["cube3.3.3"], sphere_ids["cube3.3.2"]), false, 0.0f),
    };
    // TODO - idk what is
    for (int i = 0; i < pill_flags.size(); ++i) {
        pills.push_back(std::get<0>(pill_flags[i]));
        compliances.push_back(std::get<2>(pill_flags[i]));
        if (std::get<1>(pill_flags[i])) {
            control_pills.push_back(i);
        }
    }
}

} // namespace cow