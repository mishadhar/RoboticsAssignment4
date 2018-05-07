from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy as np

def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments: 
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*

        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motionParticles = []

    #iterate thru all particles
    for part in particles:
        #get transformation based on odom
        x, y = rotate_point(odom[0], odom[1], part.h)

        #incrememnt current particle based on transformation
        part.x = part.x + x
        part.y = part.y + y

        #add translational noise for the current particle
        part.x = add_gaussian_noise(part.x, ODOM_TRANS_SIGMA)
        part.y = add_gaussian_noise(part.y, ODOM_TRANS_SIGMA)

        #increment and add noise for the rotational component of current particle
        part.h = part.h + odom[2]
        part.h = add_gaussian_noise(part.h, ODOM_HEAD_SIGMA)

        #add this particle to the returned object
        motionParticles.append(part)

    return motionParticles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments: 
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information, 
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    measuredParticles = []
    weights = []
    partsToAdd = 0

    #handle no update case first
    if len(measured_marker_list) == 0:
        return particles
    
    
    #calculate weights for each particle
    for part in particles:
        #first check if the particle is inside the grid or if the spot is occupied
        if (part.x < 0 and part.x >= grid.width and part.y < 0 and part.y >=grid.height) or ((part.x, part.y) in grid.occupied):
            #add particle with 0 probability if so and move on
            weights.append((part, 0))
            continue

        #otherwise calculate the probability for the current particle
        prob = calculate_probability(part.read_markers(grid), measured_marker_list[:])
        weights.append((part, prob))


    #sort weights by probability
    weights.sort(key=lambda x: x[1])

    #for every particle that has 0 probability (e.g. is outside the grid), increment number of new particles to add
    for part, w in weights:
        if w == 0:
            partsToAdd = partsToAdd + 1

    #swap out at least 10% of particles
    partsToAdd = max(partsToAdd, int(PARTICLE_COUNT/100))
    
    #remove 0-probability particles and add new ones
    weights = weights[partsToAdd:]
    sumProb = sum(w for part, w in weights)
    particles = [part for part, w in weights]

    #normalize weights and resample
    weights = [w/sumProb for part, w in weights]
    measuredParticles = Particle.create_random(partsToAdd, grid)
    resample = np.random.choice(particles, size=len(particles), replace=True, p=weights)

    #add noise to existing particles and append to list of newly-created random particles
    for part in resample:
        x = add_gaussian_noise(part.x, ODOM_TRANS_SIGMA)
        y = add_gaussian_noise(part.y, ODOM_TRANS_SIGMA)
        h = add_gaussian_noise(part.h, ODOM_HEAD_SIGMA)
        measuredParticles.append(Particle(x,y,h))
    return measuredParticles


#Calculates particle probability that the bot's sensors match what would be seen at given locations
def calculate_probability(particleMarkers, botMarkers):
    p = 1.0
    maxTranslationalC = 0
    maxRotationalC = (ROBOT_CAMERA_FOV_DEG**2)/(2 * (MARKER_ROT_SIGMA**2))

    #go through the max number of possible pairs between the two input lists
    for _ in range(0, min(len(particleMarkers), len(botMarkers))):
        pmout = None
        bmout = None
        nmout = None

        for bm in botMarkers:
            #inject some noise
            nm = add_marker_measurement_noise(bm, MARKER_TRANS_SIGMA, MARKER_ROT_SIGMA)
            for pm in particleMarkers:
                #check if this is the closest marker
                if pmout == None or \
                        (grid_distance(pm[0], pm[1], nm[0], nm[1]) < grid_distance(pmout[0], pmout[1], nmout[0], nmout[1])):
                    pmout = pm
                    bmout = bm
                    nmout = nm

        #remove it from the list once we find it so we don't find it again
        particleMarkers.remove(pmout)
        botMarkers.remove(bmout)

        #calculate the probability
        markerDist = grid_distance(pmout[0], pmout[1], nmout[0], nmout[1])
        markerAngle = diff_heading_deg(pmout[2], nmout[2])
        translationalC = (markerDist**2)/(2 * (MARKER_TRANS_SIGMA**2))
        rotationalC = (markerAngle**2)/(2 * (MARKER_ROT_SIGMA**2))

        p = p * math.exp(-(translationalC + rotationalC))

        maxTranslationalC = max(maxTranslationalC, translationalC)

    for _ in range(int(abs(len(particleMarkers)-len(botMarkers)))):
        p = p * math.exp(-(maxTranslationalC + maxRotationalC))

    return p