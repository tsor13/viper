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
        {"tentacle1.1", 1.000*Vec4(-23.800, -2.450, 1.280, 1.200)},
        {"tentacle1.2", 1.000*Vec4(-23.800, -2.450, -1.150, 1.200)},
        {"tentacle1.3", 1.000*Vec4(-23.800, -4.530, 0.130, 1.200)},
        {"tentacle2.1", 1.000*Vec4(-22.269, -2.374, 1.270, 1.190)},
        {"tentacle2.2", 1.000*Vec4(-22.269, -2.374, -0.909, 1.190)},
        {"tentacle2.3", 1.000*Vec4(-22.269, -4.436, 0.133, 1.190)},
        {"tentacle3.1", 1.000*Vec4(-20.738, -2.299, 1.259, 1.179)},
        {"tentacle3.2", 1.000*Vec4(-20.738, -2.299, -0.668, 1.179)},
        {"tentacle3.3", 1.000*Vec4(-20.738, -4.342, 0.136, 1.179)},
        {"tentacle4.1", 1.000*Vec4(-19.207, -2.223, 1.249, 1.169)},
        {"tentacle4.2", 1.000*Vec4(-19.207, -2.223, -0.427, 1.169)},
        {"tentacle4.3", 1.000*Vec4(-19.207, -4.248, 0.139, 1.169)},
        {"tentacle5.1", 1.000*Vec4(-17.676, -2.148, 1.239, 1.159)},
        {"tentacle5.2", 1.000*Vec4(-17.676, -2.148, -0.186, 1.159)},
        {"tentacle5.3", 1.000*Vec4(-17.676, -4.153, 0.142, 1.159)},
        {"tentacle6.1", 1.000*Vec4(-16.145, -2.072, 1.228, 1.148)},
        {"tentacle6.2", 1.000*Vec4(-16.145, -2.072, 0.055, 1.148)},
        {"tentacle6.3", 1.000*Vec4(-16.145, -4.059, 0.146, 1.148)},
        {"tentacle7.1", 1.000*Vec4(-14.614, -1.997, 1.218, 1.138)},
        {"tentacle7.2", 1.000*Vec4(-14.614, -1.997, 0.296, 1.138)},
        {"tentacle7.3", 1.000*Vec4(-14.614, -3.965, 0.149, 1.138)},
        {"tentacle8.1", 1.000*Vec4(-13.083, -1.921, 1.208, 1.128)},
        {"tentacle8.2", 1.000*Vec4(-13.083, -1.921, 0.537, 1.128)},
        {"tentacle8.3", 1.000*Vec4(-13.083, -3.871, 0.152, 1.128)},
        {"tentacle9.1", 1.000*Vec4(-11.552, -1.846, 1.197, 1.117)},
        {"tentacle9.2", 1.000*Vec4(-11.552, -1.846, 0.778, 1.117)},
        {"tentacle9.3", 1.000*Vec4(-11.552, -3.777, 0.155, 1.117)},
        {"tentacle10.1", 1.000*Vec4(-10.021, -1.770, 1.187, 1.107)},
        {"tentacle10.2", 1.000*Vec4(-10.021, -1.770, 1.019, 1.107)},
        {"tentacle10.3", 1.000*Vec4(-10.021, -3.683, 0.158, 1.107)},
        {"tentacle11.1", 1.000*Vec4(-8.414, -1.695, 1.162, 1.088)},
        {"tentacle11.2", 1.000*Vec4(-8.414, -1.695, 1.109, 1.088)},
        {"tentacle11.3", 1.000*Vec4(-8.414, -3.577, 0.151, 1.088)},
        {"tentacle12.1", 1.000*Vec4(-6.655, -1.621, 1.108, 1.052)},
        {"tentacle12.2", 1.000*Vec4(-6.655, -1.621, 0.896, 1.052)},
        {"tentacle12.3", 1.000*Vec4(-6.655, -3.449, 0.124, 1.052)},
        {"tentacle13.1", 1.000*Vec4(-4.897, -1.546, 1.054, 1.016)},
        {"tentacle13.2", 1.000*Vec4(-4.897, -1.546, 0.683, 1.016)},
        {"tentacle13.3", 1.000*Vec4(-4.897, -3.321, 0.097, 1.016)},
        {"tentacle14.1", 1.000*Vec4(-3.138, -1.472, 1.001, 0.979)},
        {"tentacle14.2", 1.000*Vec4(-3.138, -1.472, 0.470, 0.979)},
        {"tentacle14.3", 1.000*Vec4(-3.138, -3.192, 0.070, 0.979)},
        {"tentacle15.1", 1.000*Vec4(-1.379, -1.397, 0.947, 0.943)},
        {"tentacle15.2", 1.000*Vec4(-1.379, -1.397, 0.257, 0.943)},
        {"tentacle15.3", 1.000*Vec4(-1.379, -3.064, 0.043, 0.943)},
        {"tentacle16.1", 1.000*Vec4(0.379, -1.323, 0.893, 0.907)},
        {"tentacle16.2", 1.000*Vec4(0.379, -1.323, 0.043, 0.907)},
        {"tentacle16.3", 1.000*Vec4(0.379, -2.936, 0.017, 0.907)},
        {"tentacle17.1", 1.000*Vec4(2.138, -1.248, 0.839, 0.871)},
        {"tentacle17.2", 1.000*Vec4(2.138, -1.248, -0.170, 0.871)},
        {"tentacle17.3", 1.000*Vec4(2.138, -2.808, -0.010, 0.871)},
        {"tentacle18.1", 1.000*Vec4(3.897, -1.174, 0.786, 0.834)},
        {"tentacle18.2", 1.000*Vec4(3.897, -1.174, -0.383, 0.834)},
        {"tentacle18.3", 1.000*Vec4(3.897, -2.679, -0.037, 0.834)},
        {"tentacle19.1", 1.000*Vec4(5.655, -1.099, 0.732, 0.798)},
        {"tentacle19.2", 1.000*Vec4(5.655, -1.099, -0.596, 0.798)},
        {"tentacle19.3", 1.000*Vec4(5.655, -2.551, -0.064, 0.798)},
        {"tentacle20.1", 1.000*Vec4(7.414, -1.025, 0.678, 0.762)},
        {"tentacle20.2", 1.000*Vec4(7.414, -1.025, -0.809, 0.762)},
        {"tentacle20.3", 1.000*Vec4(7.414, -2.423, -0.091, 0.762)},
        {"tentacle21.1", 1.000*Vec4(9.103, -0.970, 0.609, 0.712)},
        {"tentacle21.2", 1.000*Vec4(9.103, -0.970, -0.841, 0.712)},
        {"tentacle21.3", 1.000*Vec4(9.103, -2.277, -0.101, 0.712)},
        {"tentacle22.1", 1.000*Vec4(10.759, -0.924, 0.532, 0.655)},
        {"tentacle22.2", 1.000*Vec4(10.759, -0.924, -0.782, 0.655)},
        {"tentacle22.3", 1.000*Vec4(10.759, -2.123, -0.103, 0.655)},
        {"tentacle23.1", 1.000*Vec4(12.414, -0.879, 0.456, 0.598)},
        {"tentacle23.2", 1.000*Vec4(12.414, -0.879, -0.723, 0.598)},
        {"tentacle23.3", 1.000*Vec4(12.414, -1.969, -0.106, 0.598)},
        {"tentacle24.1", 1.000*Vec4(14.069, -0.833, 0.379, 0.541)},
        {"tentacle24.2", 1.000*Vec4(14.069, -0.833, -0.664, 0.541)},
        {"tentacle24.3", 1.000*Vec4(14.069, -1.815, -0.108, 0.541)},
        {"tentacle25.1", 1.000*Vec4(15.724, -0.788, 0.303, 0.484)},
        {"tentacle25.2", 1.000*Vec4(15.724, -0.788, -0.605, 0.484)},
        {"tentacle25.3", 1.000*Vec4(15.724, -1.661, -0.110, 0.484)},
        {"tentacle26.1", 1.000*Vec4(17.379, -0.742, 0.226, 0.428)},
        {"tentacle26.2", 1.000*Vec4(17.379, -0.742, -0.546, 0.428)},
        {"tentacle26.3", 1.000*Vec4(17.379, -1.507, -0.112, 0.428)},
        {"tentacle27.1", 1.000*Vec4(19.034, -0.697, 0.150, 0.371)},
        {"tentacle27.2", 1.000*Vec4(19.034, -0.697, -0.487, 0.371)},
        {"tentacle27.3", 1.000*Vec4(19.034, -1.352, -0.114, 0.371)},
        {"tentacle28.1", 1.000*Vec4(20.690, -0.651, 0.073, 0.314)},
        {"tentacle28.2", 1.000*Vec4(20.690, -0.651, -0.428, 0.314)},
        {"tentacle28.3", 1.000*Vec4(20.690, -1.198, -0.116, 0.314)},
        {"tentacle29.1", 1.000*Vec4(22.345, -0.606, -0.003, 0.257)},
        {"tentacle29.2", 1.000*Vec4(22.345, -0.606, -0.369, 0.257)},
        {"tentacle29.3", 1.000*Vec4(22.345, -1.044, -0.118, 0.257)},
        {"tentacle30.1", 1.000*Vec4(24.000, -0.560, -0.080, 0.200)},
        {"tentacle30.2", 1.000*Vec4(24.000, -0.560, -0.310, 0.200)},
        {"tentacle30.3", 1.000*Vec4(24.000, -0.890, -0.120, 0.200)},
    };

    std::map<std::string, float> sphere_masses = {
        {"tentacle1.1", 1.440f},
        {"tentacle1.2", 1.440f},
        {"tentacle1.3", 1.440f},
        {"tentacle2.1", 1.415f},
        {"tentacle2.2", 1.415f},
        {"tentacle2.3", 1.415f},
        {"tentacle3.1", 1.391f},
        {"tentacle3.2", 1.391f},
        {"tentacle3.3", 1.391f},
        {"tentacle4.1", 1.366f},
        {"tentacle4.2", 1.366f},
        {"tentacle4.3", 1.366f},
        {"tentacle5.1", 1.342f},
        {"tentacle5.2", 1.342f},
        {"tentacle5.3", 1.342f},
        {"tentacle6.1", 1.319f},
        {"tentacle6.2", 1.319f},
        {"tentacle6.3", 1.319f},
        {"tentacle7.1", 1.295f},
        {"tentacle7.2", 1.295f},
        {"tentacle7.3", 1.295f},
        {"tentacle8.1", 1.271f},
        {"tentacle8.2", 1.271f},
        {"tentacle8.3", 1.271f},
        {"tentacle9.1", 1.248f},
        {"tentacle9.2", 1.248f},
        {"tentacle9.3", 1.248f},
        {"tentacle10.1", 1.225f},
        {"tentacle10.2", 1.225f},
        {"tentacle10.3", 1.225f},
        {"tentacle11.1", 1.184f},
        {"tentacle11.2", 1.184f},
        {"tentacle11.3", 1.184f},
        {"tentacle12.1", 1.106f},
        {"tentacle12.2", 1.106f},
        {"tentacle12.3", 1.106f},
        {"tentacle13.1", 1.031f},
        {"tentacle13.2", 1.031f},
        {"tentacle13.3", 1.031f},
        {"tentacle14.1", 0.959f},
        {"tentacle14.2", 0.959f},
        {"tentacle14.3", 0.959f},
        {"tentacle15.1", 0.889f},
        {"tentacle15.2", 0.889f},
        {"tentacle15.3", 0.889f},
        {"tentacle16.1", 0.822f},
        {"tentacle16.2", 0.822f},
        {"tentacle16.3", 0.822f},
        {"tentacle17.1", 0.758f},
        {"tentacle17.2", 0.758f},
        {"tentacle17.3", 0.758f},
        {"tentacle18.1", 0.696f},
        {"tentacle18.2", 0.696f},
        {"tentacle18.3", 0.696f},
        {"tentacle19.1", 0.637f},
        {"tentacle19.2", 0.637f},
        {"tentacle19.3", 0.637f},
        {"tentacle20.1", 0.581f},
        {"tentacle20.2", 0.581f},
        {"tentacle20.3", 0.581f},
        {"tentacle21.1", 0.507f},
        {"tentacle21.2", 0.507f},
        {"tentacle21.3", 0.507f},
        {"tentacle22.1", 0.429f},
        {"tentacle22.2", 0.429f},
        {"tentacle22.3", 0.429f},
        {"tentacle23.1", 0.358f},
        {"tentacle23.2", 0.358f},
        {"tentacle23.3", 0.358f},
        {"tentacle24.1", 0.293f},
        {"tentacle24.2", 0.293f},
        {"tentacle24.3", 0.293f},
        {"tentacle25.1", 0.235f},
        {"tentacle25.2", 0.235f},
        {"tentacle25.3", 0.235f},
        {"tentacle26.1", 0.183f},
        {"tentacle26.2", 0.183f},
        {"tentacle26.3", 0.183f},
        {"tentacle27.1", 0.137f},
        {"tentacle27.2", 0.137f},
        {"tentacle27.3", 0.137f},
        {"tentacle28.1", 0.098f},
        {"tentacle28.2", 0.098f},
        {"tentacle28.3", 0.098f},
        {"tentacle29.1", 0.066f},
        {"tentacle29.2", 0.066f},
        {"tentacle29.3", 0.066f},
        {"tentacle30.1", 0.040f},
        {"tentacle30.2", 0.040f},
        {"tentacle30.3", 0.040f},
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
        tuple(Vec2i(sphere_ids["tentacle1.1"], sphere_ids["tentacle1.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle1.1"], sphere_ids["tentacle1.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle1.2"], sphere_ids["tentacle1.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle2.1"], sphere_ids["tentacle2.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle2.1"], sphere_ids["tentacle2.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle2.2"], sphere_ids["tentacle2.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle3.1"], sphere_ids["tentacle3.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle3.1"], sphere_ids["tentacle3.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle3.2"], sphere_ids["tentacle3.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle4.1"], sphere_ids["tentacle4.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle4.1"], sphere_ids["tentacle4.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle4.2"], sphere_ids["tentacle4.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle5.1"], sphere_ids["tentacle5.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle5.1"], sphere_ids["tentacle5.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle5.2"], sphere_ids["tentacle5.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle6.1"], sphere_ids["tentacle6.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle6.1"], sphere_ids["tentacle6.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle6.2"], sphere_ids["tentacle6.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle7.1"], sphere_ids["tentacle7.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle7.1"], sphere_ids["tentacle7.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle7.2"], sphere_ids["tentacle7.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle8.1"], sphere_ids["tentacle8.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle8.1"], sphere_ids["tentacle8.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle8.2"], sphere_ids["tentacle8.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle9.1"], sphere_ids["tentacle9.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle9.1"], sphere_ids["tentacle9.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle9.2"], sphere_ids["tentacle9.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle10.1"], sphere_ids["tentacle10.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle10.1"], sphere_ids["tentacle10.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle10.2"], sphere_ids["tentacle10.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle11.1"], sphere_ids["tentacle11.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle11.1"], sphere_ids["tentacle11.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle11.2"], sphere_ids["tentacle11.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle12.1"], sphere_ids["tentacle12.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle12.1"], sphere_ids["tentacle12.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle12.2"], sphere_ids["tentacle12.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle13.1"], sphere_ids["tentacle13.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle13.1"], sphere_ids["tentacle13.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle13.2"], sphere_ids["tentacle13.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle14.1"], sphere_ids["tentacle14.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle14.1"], sphere_ids["tentacle14.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle14.2"], sphere_ids["tentacle14.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle15.1"], sphere_ids["tentacle15.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle15.1"], sphere_ids["tentacle15.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle15.2"], sphere_ids["tentacle15.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle16.1"], sphere_ids["tentacle16.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle16.1"], sphere_ids["tentacle16.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle16.2"], sphere_ids["tentacle16.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle17.1"], sphere_ids["tentacle17.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle17.1"], sphere_ids["tentacle17.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle17.2"], sphere_ids["tentacle17.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle18.1"], sphere_ids["tentacle18.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle18.1"], sphere_ids["tentacle18.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle18.2"], sphere_ids["tentacle18.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle19.1"], sphere_ids["tentacle19.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle19.1"], sphere_ids["tentacle19.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle19.2"], sphere_ids["tentacle19.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle20.1"], sphere_ids["tentacle20.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle20.1"], sphere_ids["tentacle20.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle20.2"], sphere_ids["tentacle20.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle21.1"], sphere_ids["tentacle21.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle21.1"], sphere_ids["tentacle21.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle21.2"], sphere_ids["tentacle21.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle22.1"], sphere_ids["tentacle22.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle22.1"], sphere_ids["tentacle22.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle22.2"], sphere_ids["tentacle22.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle23.1"], sphere_ids["tentacle23.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle23.1"], sphere_ids["tentacle23.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle23.2"], sphere_ids["tentacle23.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle24.1"], sphere_ids["tentacle24.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle24.1"], sphere_ids["tentacle24.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle24.2"], sphere_ids["tentacle24.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle25.1"], sphere_ids["tentacle25.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle25.1"], sphere_ids["tentacle25.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle25.2"], sphere_ids["tentacle25.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle26.1"], sphere_ids["tentacle26.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle26.1"], sphere_ids["tentacle26.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle26.2"], sphere_ids["tentacle26.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle27.1"], sphere_ids["tentacle27.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle27.1"], sphere_ids["tentacle27.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle27.2"], sphere_ids["tentacle27.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle28.1"], sphere_ids["tentacle28.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle28.1"], sphere_ids["tentacle28.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle28.2"], sphere_ids["tentacle28.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle29.1"], sphere_ids["tentacle29.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle29.1"], sphere_ids["tentacle29.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle29.2"], sphere_ids["tentacle29.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle30.1"], sphere_ids["tentacle30.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle30.1"], sphere_ids["tentacle30.3"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle30.2"], sphere_ids["tentacle30.3"]), true, 0.0001f),


        tuple(Vec2i(sphere_ids["tentacle1.1"], sphere_ids["tentacle2.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle1.2"], sphere_ids["tentacle2.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle1.3"], sphere_ids["tentacle2.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle2.1"], sphere_ids["tentacle3.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle2.2"], sphere_ids["tentacle3.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle2.3"], sphere_ids["tentacle3.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle3.1"], sphere_ids["tentacle4.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle3.2"], sphere_ids["tentacle4.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle3.3"], sphere_ids["tentacle4.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle4.1"], sphere_ids["tentacle5.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle4.2"], sphere_ids["tentacle5.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle4.3"], sphere_ids["tentacle5.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle5.1"], sphere_ids["tentacle6.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle5.2"], sphere_ids["tentacle6.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle5.3"], sphere_ids["tentacle6.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle6.1"], sphere_ids["tentacle7.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle6.2"], sphere_ids["tentacle7.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle6.3"], sphere_ids["tentacle7.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle7.1"], sphere_ids["tentacle8.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle7.2"], sphere_ids["tentacle8.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle7.3"], sphere_ids["tentacle8.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle8.1"], sphere_ids["tentacle9.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle8.2"], sphere_ids["tentacle9.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle8.3"], sphere_ids["tentacle9.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle9.1"], sphere_ids["tentacle10.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle9.2"], sphere_ids["tentacle10.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle9.3"], sphere_ids["tentacle10.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle10.1"], sphere_ids["tentacle11.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle10.2"], sphere_ids["tentacle11.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle10.3"], sphere_ids["tentacle11.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle11.1"], sphere_ids["tentacle12.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle11.2"], sphere_ids["tentacle12.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle11.3"], sphere_ids["tentacle12.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle12.1"], sphere_ids["tentacle13.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle12.2"], sphere_ids["tentacle13.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle12.3"], sphere_ids["tentacle13.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle13.1"], sphere_ids["tentacle14.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle13.2"], sphere_ids["tentacle14.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle13.3"], sphere_ids["tentacle14.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle14.1"], sphere_ids["tentacle15.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle14.2"], sphere_ids["tentacle15.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle14.3"], sphere_ids["tentacle15.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle15.1"], sphere_ids["tentacle16.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle15.2"], sphere_ids["tentacle16.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle15.3"], sphere_ids["tentacle16.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle16.1"], sphere_ids["tentacle17.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle16.2"], sphere_ids["tentacle17.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle16.3"], sphere_ids["tentacle17.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle17.1"], sphere_ids["tentacle18.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle17.2"], sphere_ids["tentacle18.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle17.3"], sphere_ids["tentacle18.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle18.1"], sphere_ids["tentacle19.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle18.2"], sphere_ids["tentacle19.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle18.3"], sphere_ids["tentacle19.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle19.1"], sphere_ids["tentacle20.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle19.2"], sphere_ids["tentacle20.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle19.3"], sphere_ids["tentacle20.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle20.1"], sphere_ids["tentacle21.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle20.2"], sphere_ids["tentacle21.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle20.3"], sphere_ids["tentacle21.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle21.1"], sphere_ids["tentacle22.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle21.2"], sphere_ids["tentacle22.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle21.3"], sphere_ids["tentacle22.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle22.1"], sphere_ids["tentacle23.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle22.2"], sphere_ids["tentacle23.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle22.3"], sphere_ids["tentacle23.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle23.1"], sphere_ids["tentacle24.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle23.2"], sphere_ids["tentacle24.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle23.3"], sphere_ids["tentacle24.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle24.1"], sphere_ids["tentacle25.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle24.2"], sphere_ids["tentacle25.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle24.3"], sphere_ids["tentacle25.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle25.1"], sphere_ids["tentacle26.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle25.2"], sphere_ids["tentacle26.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle25.3"], sphere_ids["tentacle26.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle26.1"], sphere_ids["tentacle27.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle26.2"], sphere_ids["tentacle27.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle26.3"], sphere_ids["tentacle27.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle27.1"], sphere_ids["tentacle28.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle27.2"], sphere_ids["tentacle28.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle27.3"], sphere_ids["tentacle28.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle28.1"], sphere_ids["tentacle29.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle28.2"], sphere_ids["tentacle29.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle28.3"], sphere_ids["tentacle29.3"]), true, 0.0001f),

        tuple(Vec2i(sphere_ids["tentacle29.1"], sphere_ids["tentacle30.1"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle29.2"], sphere_ids["tentacle30.2"]), true, 0.0001f),
        tuple(Vec2i(sphere_ids["tentacle29.3"], sphere_ids["tentacle30.3"]), true, 0.0001f),

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