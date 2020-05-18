import os
import numpy as np

filename = 'Octopus.h'

num_joints = 30
scale = 1
inter_compliance = 1e-4
other_compliance = 1e-4

cube_radius = 1
cube_center = np.array([4.0, 4.0, 4.0])
cube_n = 4
cube_component_radius = cube_radius / cube_n

# tentacle depth, tentacle part, (x, y, z, r)
annotated = np.array([ [ [-23.8, -2.45, 1.28, 1.2],
                         [-23.8, -2.45, -1.15, 1.2],
                         [-23.8, -4.53, 0.13, 1.2] ],

                       [ [-9.0, -1.72, 1.18, 1.1],
                         [-9.0, -1.72, 1.18, 1.1],
                         [-9.0, -3.62, 0.16, 1.1] ],

                       [ [8.0, -1.0, 0.66, 0.75],
                         [8.0, -1.0, -0.88, 0.75],
                         [8.0, -2.38, -.1, 0.75] ],

                       [ [24, -.56, -.08, .2],
                         [24, -.56, -.31, .2],
                         [24, -.89, -.12, .2] ] ])

def interpolate(x):
    if x == 1:
        return annotated[-1]
    fractions = len(annotated) - 1
    i = 0
    while x - i/fractions > 1/fractions:
        i += 1
    x = (x - i/fractions) * fractions
    # convex combination
    return x * annotated[i+1] + (1-x) * annotated[i]

joints = np.array([interpolate(x) for x in np.linspace(0, 1, num_joints)])


head = '''// Licensed under the Apache License, Version 2.0 (the "License");
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
    std::map<std::string, Vec4> sphere_vals = {'''


mid = '''    // make ids for each sphere
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
    std::vector<tuple> pill_flags = {'''


end = '''    // TODO - idk what is
    for (int i = 0; i < pill_flags.size(); ++i) {
        pills.push_back(std::get<0>(pill_flags[i]));
        compliances.push_back(std::get<2>(pill_flags[i]));
        if (std::get<1>(pill_flags[i])) {
            control_pills.push_back(i);
        }
    }
}

} // namespace cow'''

with open(filename, 'w') as f:
    # write head
    f.write(head)

    # write Sphere vals
    for i in range(joints.shape[0]):
        for j in range(joints.shape[1]):
            f.write('\n        {')
            f.write('"tentacle{}.{}", {:.3f}*Vec4({:.3f}, {:.3f}, {:.3f}, {:.3f})'.format(i+1, j+1, scale, *joints[i,j]))
            f.write('},')
    f.write('\n')
    # write Cube vals
    deltas = np.linspace(-cube_radius + cube_component_radius, cube_radius - cube_component_radius, cube_n)
    for i, dx in enumerate(deltas):
        for j, dy in enumerate(deltas):
            for k, dz in enumerate(deltas):
                f.write('\n        {')
                delta = np.array([dx, dy, dz])
                p = cube_center + delta
                f.write('"cube{}.{}.{}", {:.3f}*Vec4({:.3f}, {:.3f}, {:.3f}, {:.3f})'.format(i+1, j+1, k+1, scale, *p, cube_component_radius))
                f.write('},')
    f.write('\n    };\n')

    # write masses
    f.write('\n    std::map<std::string, float> sphere_masses = {')
    for i in range(joints.shape[0]):
        for j in range(joints.shape[1]):
            f.write('\n        {')
            # mass is area (squared) of radius
            f.write('"tentacle{}.{}", {:.3f}f'.format(i+1, j+1, (scale*joints[i,j,-1])**2))
            f.write('},')
    f.write('\n')
    cube_mass = (scale*cube_component_radius)**2
    for i in range(cube_n):
        for j in range(cube_n):
            for k in range(cube_n):
                f.write('\n        {')
                # mass is area (squared) of radius
                f.write('"cube{}.{}.{}", {:.3f}f'.format(i+1, j+1, k+1, cube_mass))
                f.write('},')
    f.write('\n    };\n')
    f.write(mid)

    # write pill connections
    # level to level conections
    # 3n connections
    for i in range(joints.shape[0]):
        for j in range(joints.shape[1]):
            if i+1 < joints.shape[0]:
                f.write('\n        tuple(Vec2i(sphere_ids["tentacle{}.{}"], sphere_ids["tentacle{}.{}"]), true, {}f),'.format(i+1, j+1, i+2, j+1, other_compliance))
        f.write('\n')
    # inter-level connections
    f.write('\n')
    # 3n connections
    for i in range(joints.shape[0]):
        for j in range(joints.shape[1]):
            for k in range(j+1, joints.shape[1]):
                f.write('\n        tuple(Vec2i(sphere_ids["tentacle{}.{}"], sphere_ids["tentacle{}.{}"]), true, {}f),'.format(i+1, j+1, i+1, k+1, inter_compliance))
        f.write('\n')

    def in_bounds(i):
        return 0 <= i <= cube_n-1
    for i in range(cube_n):
        for j in range(cube_n):
            for k in range(cube_n):
                for i2 in [i-1, i+1]:
                    if in_bounds(i2):
                        f.write('\n        ')
                        f.write('tuple(Vec2i(sphere_ids["cube{}.{}.{}"], sphere_ids["cube{}.{}.{}"]), true, 0.0f),'.format(i+1, j+1, k+1, i2+1, j+1, k+1))
                for j2 in [j-1, j+1]:
                    if in_bounds(j2):
                        f.write('\n        ')
                        f.write('tuple(Vec2i(sphere_ids["cube{}.{}.{}"], sphere_ids["cube{}.{}.{}"]), true, 0.0f),'.format(i+1, j+1, k+1, i+1, j2+1, k+1))
                for k2 in [k-1, k+1]:
                    if in_bounds(k2):
                        f.write('\n        ')
                        f.write('tuple(Vec2i(sphere_ids["cube{}.{}.{}"], sphere_ids["cube{}.{}.{}"]), true, 0.0f),'.format(i+1, j+1, k+1, i+1, j+1, k2+1))

    f.write('\n    };\n')
    f.write(end)
