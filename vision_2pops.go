package main
// Compiled to evolution_sim.out
import (
	"fmt"
	"flag"
	"strconv"
	"math/rand"
	"math"
	"time"
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

	//set up seed
	rand.Seed(time.Now().UTC().UnixNano())

	// Reading in parameters
	rescDensityParam := flag.String("rescDensity", "0.75", "Resource Per Unit Area")
	rescPeriodParam := flag.Int("rescPeriod", 1, "Ticks till new resources added")
	energyQuantParam := flag.String("energyQuantity", "1.0", "The Energy per resource location")
	initPopSizeParam := flag.Int("initPopSize", 100, "The initial population of agents")
	initRadiusParam := flag.String("initRadius", "0.0", "The initial perception radius of agents")
	basalCostParam := flag.String("basalEnergyCost", "0.05", "The basal energy cost per tick")
	radiusCostParam := flag.String("radiusCost", "0.1", "Energy cost per unit perceptual radius")
	reproCostParam := flag.String("reproCost", "0.5", "Energy cost to reproduce")
	growthRateParam := flag.String("growthRate", "0.5", "Probability to reproduce each tick")
	maxMutateParam := flag.String("maxMutate", "0.25", "Maximum mutation per reproduction")
	agentSpeedParam := flag.String("agentSpeed", "1.0", "Speed of agents in unit space/time")
	gatherDistParam := flag.String("gatherDist", "1.0", "The distance with which an agent can gather resource")
	gatherAmountParam := flag.String("gatherAmount", "0.5", "Energy gathered when agent gathers from a resource")
	itersParam := flag.Int("iters", 150, "Number of iterations to run the simulation")
	maxPopParam := flag.Int("maxPop", 100000, "Maximum agent population")
	widthParam := flag.Int("width", 100, "Width of the environment")
	heightParam := flag.Int("height", 100, "Height of the environment")
	resourceMvtSpeedParam := flag.Float64("resourceMovementSpeed", 0.0, "Speed with which resources randomly move.")
	verboseParam := flag.Bool("verbose", true, "print details?")
	energyCapParam := flag.String("energyCap", "2.0", "cap on energy an agent can store")
	percentageParam := flag.String("percDec", "0.5", "Decimal (percent) of population with nonzero perceptual range")
	fixedRangeParam := flag.String("fixedRange", "0.25", "Fixed perceptual range for the population that has a perceptual range")

	flag.Parse()

	// Process params into variables
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
	resourceMvtSpeed := *(resourceMvtSpeedParam)
	verbose := *(verboseParam)
	percentage,_ := strconv.ParseFloat((*percentageParam), 64)
	fixedRange,_ := strconv.ParseFloat((*fixedRangeParam), 64)
	energyCap,_ := strconv.ParseFloat((*energyCapParam), 64)

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
		fmt.Printf("# percentage: %f\n", percentage)
		fmt.Printf("# fixedRange: %f\n", fixedRange)
	}

	// running the simulation
	// build the initial populations
	fixedRangePopSize := int(percentage * float64(initPopSize))
	zeroRangePopSize := initPopSize - fixedRangePopSize
	agentList := []*Agent{}
	resourceList := []*Resource{}
	for i:=0; i < fixedRangePopSize; i++ {
		a := Agent{}
		a.X = floatBetween(0.0, float64(width))
		a.Y = floatBetween(0.0, float64(height))
		a.Radius = fixedRange
		a.Speed = agentSpeed
		a.Energy = 1.0
		agentList = append(agentList, &a)
	}
	for i:=0; i < zeroRangePopSize; i++ {
		a := Agent{}
		a.X = floatBetween(0.0, float64(width))
		a.Y = floatBetween(0.0, float64(height))
		a.Radius = 0
		a.Speed = agentSpeed
		a.Energy = 1.0
		agentList = append(agentList, &a)
	}

	// set up initial resources
	rescToAdd := int(math.Floor(rescDensity*float64(width)*float64(height)))
	initTotalRescQuant := float64(rescToAdd)*energyQuant
	for i:=0; i<rescToAdd; i++ {
		r := Resource{}
		r.X = floatBetween(0.0, float64(width))
		r.Y = floatBetween(0.0, float64(height))
		r.Quantity = energyQuant
		resourceList = append(resourceList, &r)
	}

	// run the simulation for the preset amount of iterations
	// print header of output
	header := "Iteration,Type,Energy,Radius\n"
	fmt.Printf(header)
	for iter := 0; iter < iters; iter++ {

		offSpringList := []*Agent{}

		// write output
            // EDIT: prints 0 in second column if agent, 1 for resource
            // EDIT 2: uses "," instead of "|" for delimiter, so the output can go to .csv files
		for i:=0; i < len(agentList); i++ {
			a := agentList[i]
			fmt.Printf("%d,0,%f,%f\n", iter, a.Energy, a.Radius)
		}
		for i:=0; i < len(resourceList); i++ {
			r := resourceList[i]
			fmt.Printf("%d,1,%f,%f\n", iter, r.Quantity, 0.0)
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

			// check agent position within bounds
			// torus boundary condition
			if(a.X < 0.0) {
				a.X = a.X + float64(width)
			}
			if(a.X >= float64(width)) {
				a.X = a.X - float64(width)
			}
			if(a.Y < 0.0) {
				a.Y = a.Y + float64(height)
			}
			if(a.Y >= float64(height)) {
				a.Y = a.Y - float64(height)
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
				if a.Energy > energyCap {
					a.Energy = energyCap
				}
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
				newAgent.Radius = a.Radius
				newAgent.Speed = a.Speed
				offSpringList = append(offSpringList, &newAgent)
				a.Energy -= reproCost // take reproduction penalty
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
		// update resources resources
		tempR := []*Resource{}
		rescLeft := 0.0
		for i:=0; i < len(resourceList); i++ {

			//move resources
			resc := resourceList[i]
			theta := floatBetween(0.0, 2*math.Pi)
			dx := resourceMvtSpeed*math.Cos(theta)
			dy := resourceMvtSpeed*math.Sin(theta)
			resc.X += dx
			resc.Y += dy
			//check boundaries
			if(resc.X < 0.0) {
				resc.X = resc.X + float64(width)
			}
			if(resc.X >= float64(width)) {
				resc.X = resc.X - float64(width)
			}
			if(resc.Y < 0.0) {
				resc.Y = resc.Y + float64(height)
			}
			if(resc.Y >= float64(height)) {
				resc.Y = resc.Y - float64(height)
			}

			if(resourceList[i].Quantity > 0.0) {
				tempR = append(tempR, resourceList[i])
				rescLeft += resourceList[i].Quantity
			}
		}
		resourceList = tempR
		// add new resources if neccessary
		if(iter%rescPeriod == 0 && iter != 0) {
			totalRescToAdd := initTotalRescQuant - rescLeft
			rescToAdd := int(math.Floor(totalRescToAdd/energyQuant))
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


