package main

import (
	"fmt"
	"flag"
	"strconv"
	"math/rand"
	"math"
)

/**
The Agent struct instantiates a single agent in the environment
**/
type Agent struct {
	Radius float64 //The perception radius of the agent
	X float64 //The x coordinate of the agent
	Y float64 //The y coordinate of the agent
	Speed float64 //The speed with which the agent moves (Scalar)
	Energy float64 //The resevoir of energy for this agent
}

type Resource struct {
	Quantity float64 //The amount of resource here
	X float64 //The x coordinate of the resource
	Y float64 //The y coordinate of the resource
}

func floatBetween(a float64, b float64) float64 {
	r := rand.Float64()
	r *= (b - a)
	r += a
	return r
}

//euclidean distance between points (ax, ay) and (bx, by)
func euclidDist(ax float64, ay float64, bx float64, by float64) float64 {
	return math.Sqrt((ax-bx)*(ax-bx) + (ay-by)*(ay-by))
}

// Main function
func main() {
	// Reading in parameters
	rescDensityParam := flag.String("rescDensity", "0.1", "Resource Per Unit Area")
	rescPeriodParam := flag.Int("rescPeriod", 100, "Ticks till new resources added")
	energyQuantParam := flag.String("energyQuantity", "1.0", "The Energy per resource location")
	initPopSizeParam := flag.Int("initPopSize", 10, "The initial population of agents")
	initRadiusParam := flag.String("initRadius", "0.0", "The initial perception radius of agents")
	basalCostParam := flag.String("basalEnergyCost", "0.1", "The basal energy cost per tick")
	radiusCostParam := flag.String("radiusCost", "0.1", "Energy cost per unit perceptual radius")
	reproCostParam := flag.String("reproCost", "0.1", "Energy cost to reproduce")
	growthRateParam := flag.String("growthRate", "0.01", "Probability to reproduce each tick")
	maxMutateParam := flag.String("maxMutate", "0.25", "Maximum mutation per reproduction")
	agentSpeedParam := flag.String("agentSpeed", "1.0", "Speed of agents in unit space/time")
	gatherDistParam := flag.String("gatherDist", "1.0", "The distance with which an agent can gather resource")
	gatherAmountParam := flag.String("gatherAmount", "0.1", "Energy gathered when agent gathers from a resource")
	itersParam := flag.Int("iters", 100, "Number of iterations to run the simulation")
	maxPopParam := flag.Int("maxPop", 10000, "Maximum agent population")
	widthParam := flag.Int("width", 100, "Width of the environment")
	heightParam := flag.Int("height", 100, "Height of the environment")
	verboseParam := flag.Bool("verbose", true, "print details?")

	flag.Parse()

	//Process params into variables
	rescDensity,_ := strconv.ParseFloat((*rescDensityParam), 64)
	rescPeriod := *(rescPeriodParam)
	energyQuant,_ := strconv.ParseFloat((*energyQuantParam), 64)
	initPopSize := *(initPopSizeParam)
	initRadius,_ := strconv.ParseFloat((*initRadiusParam), 64)
	radiusCost,_ := strconv.ParseFloat((*radiusCostParam), 64)
	reproCost,_ := strconv.ParseFloat((*reproCostParam), 64)
	growthRate,_ := strconv.ParseFloat((*growthRateParam), 64)
	maxMutate,_ := strconv.ParseFloat((*maxMutateParam), 64)
	agentSpeed,_ := strconv.ParseFloat((*agentSpeedParam), 64)
	gatherDist,_ := strconv.ParseFloat((*gatherDistParam), 64)
	basalCost,_ := strconv.ParseFloat((*basalCostParam), 64)
	gatherAmount,_ := strconv.ParseFloat((*gatherAmountParam), 64)
	width := *(widthParam)
	height := *(heightParam)
	iters := *(itersParam)
	maxPop := *(maxPopParam)
	verbose := *(verboseParam)

	if(verbose) {
		fmt.Printf("# rescDensity: %f\n", rescDensity)
		fmt.Printf("# rescPeriod: %d\n", rescPeriod)
		fmt.Printf("# energyQuant: %f\n", energyQuant)
		fmt.Printf("# initPopSize: %d\n", initPopSize)
		fmt.Printf("# initRadius: %f\n", initRadius)
		fmt.Printf("# radiusCost: %f\n", radiusCost)
		fmt.Printf("# reproCost: %f\n", reproCost)
		fmt.Printf("# growthRate: %f\n", growthRate)
		fmt.Printf("# maxMutate: %f\n", maxMutate)
		fmt.Printf("# agentSpeed: %f\n", agentSpeed)
		fmt.Printf("# gatherDist: %f\n", gatherDist)
		fmt.Printf("# gatherAmount: %f\n", gatherAmount)
		fmt.Printf("# basalCost: %f\n", basalCost)
		fmt.Printf("# iters: %d\n", iters)
		fmt.Printf("# Width: %d\n", width)
		fmt.Printf("# Height: %d\n", height)
		fmt.Printf("# maxPop: %d\n", maxPop)
	}

	// running the simulation
	// build the initial population
	agentList := []*Agent{}
	resourceList := []*Resource{}
	for i:=0; i < initPopSize; i++ {
		a := Agent{}
		a.X = floatBetween(0.0, float64(width))
		a.Y = floatBetween(0.0, float64(height))
		a.Radius = initRadius
		a.Speed = agentSpeed
		a.Energy = 1.0
		agentList = append(agentList, &a)
	}
	// set up initial resources
	rescToAdd := int(math.Floor(rescDensity*float64(width)*float64(height)))
	for i:=0; i<rescToAdd; i++ {
		r := Resource{}
		r.X = floatBetween(0.0, float64(width))
		r.Y = floatBetween(0.0, float64(height))
		r.Quantity = energyQuant
		resourceList = append(resourceList, &r)
	}
	// run the simulation for the preset amount of iterations
	// print header of output
	header := "Iteration|Type|X|Y|Energy|Radius\n"
	fmt.Printf(header)
	for iter := 0; iter < iters; iter++ {

		offSpringList := []*Agent{}

		// write output
		for i:=0; i < len(agentList); i++ {
			a := agentList[i]
			fmt.Printf("%d|agent|%f|%f|%f|%f\n", iter, a.X, a.Y, a.Energy, a.Radius)
		}
		for i:=0; i < len(resourceList); i++ {
			r := resourceList[i]
			fmt.Printf("%d|resource|%f|%f|%f|%f\n", iter, r.X, r.Y, r.Quantity, 0.0)
		}
		// update agents
		for i:=0; i < len(agentList); i++ {
			//find distance to closest resource if resources exist
			a := agentList[i]
			var closestResource *Resource
			hasClosestResource := false
			resourceInGatherDist := false
			closestDist := 0.0
			if(len(resourceList) > 0) {
				for j:=0; j < len(resourceList); j++ {
					r := resourceList[j]
					d := euclidDist(a.X, a.Y, r.X, r.Y)
					if(d < a.Radius && !hasClosestResource) {
						hasClosestResource = true
						closestDist = d
						closestResource = r
					} else if (d < a.Radius && d < closestDist) {
						closestDist = d
						closestResource = r
					}
				}
				if(closestDist < gatherDist && hasClosestResource) {
					resourceInGatherDist = true
				}
			}
			// diffuse randomly if no nearby resource
			if(!hasClosestResource) {
				theta := floatBetween(0.0, 2*math.Pi)
				dx := a.Speed*math.Cos(theta)
				dy := a.Speed*math.Sin(theta)
				a.X += dx
				a.Y += dy
			} else { //move toward sensed resource
				theta := math.Atan2((closestResource.Y - a.Y), (closestResource.X - a.X))
				dx := a.Speed*math.Cos(theta)
				dy := a.Speed*math.Sin(theta)
				a.X += dx
				a.Y += dy
			}
			// gather if possible
			if(resourceInGatherDist) {
				amount := gatherAmount
				if(closestResource.Quantity < gatherAmount) {
					amount = closestResource.Quantity
					if(amount <= 0.0) {
						amount = 0.0
					}
				}
				a.Energy += amount
				closestResource.Quantity -= amount
			}
			// take cost penalty
			cost := basalCost + radiusCost*a.Radius
			a.Energy -= cost

			// decide whether or not to reproduce
			if(rand.Float64() < growthRate && a.Energy >= reproCost) {
				newAgent := Agent{}
				newAgent.X = a.X + rand.Float64()*5.0 - 2.5
				newAgent.Y = a.Y + rand.Float64()*5.0 - 2.5
				newAgent.Energy = 1.0
				newAgent.Radius = a.Radius + floatBetween(0.0, 2*maxMutate) - maxMutate
				if(newAgent.Radius <= 0.0) {
					newAgent.Radius = 0.0
				}
				newAgent.Speed = a.Speed
				offSpringList = append(offSpringList, &newAgent)
				// take reproduction penalty
				a.Energy -= reproCost
			}

		}

		// remove dead agents
		temp := []*Agent{}
		for i:=0; i < len(agentList); i++ {
			if agentList[i].Energy >= 0.0 {
				temp = append(temp, agentList[i])
			}
		}
		//add new agents
		for i:=0; i < len(offSpringList); i++ {
			if(len(temp) < maxPop) {
				temp = append(temp, offSpringList[i])
			}
		}
		agentList = temp
		// remove empty resources
		tempR := []*Resource{}
		for i:=0; i < len(resourceList); i++ {
			if(resourceList[i].Quantity > 0.0) {
				tempR = append(tempR, resourceList[i])
			}
		}
		resourceList = tempR
		// add new resources if neccessary
		if(iter%rescPeriod == 0 && iter != 0) {
			for i:=0; i < rescToAdd; i++ {
				r := Resource{}
				r.X = floatBetween(0.0, float64(width))
				r.Y = floatBetween(0.0, float64(height))
				r.Quantity = energyQuant
				resourceList = append(resourceList, &r)
			}
		}

	}
}


