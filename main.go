package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
	"github.com/go-echarts/go-echarts/v2/types"
)

// Wine represents a single wine sample with its features and quality
type Wine struct {
	features []float64
	quality  float64
}

// LinearRegression represents our multiple linear regression model
type LinearRegression struct {
	weights     []float64
	bias        float64
	numFeatures int
	means       []float64
	stds        []float64
}

// NewLinearRegression creates a new linear regression model
func NewLinearRegression(numFeatures int) *LinearRegression {
	return &LinearRegression{
		weights:     make([]float64, numFeatures),
		bias:        0.0,
		numFeatures: numFeatures,
		means:       make([]float64, numFeatures),
		stds:        make([]float64, numFeatures),
	}
}

// normalizeFeatures normalizes the features using z-score normalization
func (lr *LinearRegression) normalizeFeatures(data []Wine) []Wine {
	// Calculate means and standard deviations
	for i := 0; i < lr.numFeatures; i++ {
		sum := 0.0
		for _, wine := range data {
			sum += wine.features[i]
		}
		lr.means[i] = sum / float64(len(data))

		sumSquares := 0.0
		for _, wine := range data {
			sumSquares += math.Pow(wine.features[i]-lr.means[i], 2)
		}
		lr.stds[i] = math.Sqrt(sumSquares / float64(len(data)))
		if lr.stds[i] == 0 {
			lr.stds[i] = 1 // Prevent division by zero
		}
	}

	// Normalize the features
	normalizedData := make([]Wine, len(data))
	for i, wine := range data {
		normalizedFeatures := make([]float64, lr.numFeatures)
		for j := 0; j < lr.numFeatures; j++ {
			normalizedFeatures[j] = (wine.features[j] - lr.means[j]) / lr.stds[j]
		}
		normalizedData[i] = Wine{
			features: normalizedFeatures,
			quality:  wine.quality,
		}
	}
	return normalizedData
}

// predict makes a prediction for a single wine sample
func (lr *LinearRegression) predict(features []float64) float64 {
	// Normalize the input features
	normalizedFeatures := make([]float64, len(features))
	for i := 0; i < len(features); i++ {
		normalizedFeatures[i] = (features[i] - lr.means[i]) / lr.stds[i]
	}

	prediction := lr.bias
	for i := 0; i < lr.numFeatures; i++ {
		prediction += lr.weights[i] * normalizedFeatures[i]
	}
	return prediction
}

// train trains the model using gradient descent
func (lr *LinearRegression) train(data []Wine, learningRate float64, epochs int) []float64 {
	mseHistory := make([]float64, epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0

		// Shuffle the data for each epoch
		rand.Shuffle(len(data), func(i, j int) {
			data[i], data[j] = data[j], data[i]
		})

		for _, wine := range data {
			prediction := lr.predict(wine.features)
			error := prediction - wine.quality

			// Update bias
			lr.bias -= learningRate * error

			// Update weights
			for i := 0; i < lr.numFeatures; i++ {
				lr.weights[i] -= learningRate * error * wine.features[i]
			}

			totalError += error * error
		}

		// Calculate MSE for this epoch
		mse := totalError / float64(len(data))
		mseHistory[epoch] = mse

		if epoch%100 == 0 {
			fmt.Printf("Epoch %d - MSE: %.4f\n", epoch, mse)
		}
	}
	return mseHistory
}

// crossValidate performs k-fold cross-validation
func crossValidate(data []Wine, k int, learningRate float64, epochs int) []float64 {
	foldSize := len(data) / k
	scores := make([]float64, k)

	for i := 0; i < k; i++ {
		// Create validation fold
		start := i * foldSize
		end := start + foldSize
		validationData := append([]Wine{}, data[start:end]...)
		trainingData := append(append([]Wine{}, data[:start]...), data[end:]...)

		// Create and train model
		model := NewLinearRegression(len(data[0].features))
		normalizedTrainingData := model.normalizeFeatures(trainingData)
		model.train(normalizedTrainingData, learningRate, epochs)

		// Evaluate on validation fold
		scores[i] = calculateR2(model, validationData)
	}
	return scores
}

// createLearningCurvePlot creates a line chart of the learning curve
func createLearningCurvePlot(mseHistory []float64) {
	line := charts.NewLine()
	line.SetGlobalOptions(
		charts.WithTitleOpts(opts.Title{
			Title: "Learning Curve",
		}),
		charts.WithInitializationOpts(opts.Initialization{
			Theme: types.ThemeWesteros,
		}),
		charts.WithXAxisOpts(opts.XAxis{
			Name: "Epoch",
		}),
		charts.WithYAxisOpts(opts.YAxis{
			Name: "Mean Squared Error",
		}),
	)

	// Prepare data
	xAxis := make([]int, len(mseHistory))
	yAxis := make([]opts.LineData, len(mseHistory))
	for i := range mseHistory {
		xAxis[i] = i
		yAxis[i] = opts.LineData{Value: mseHistory[i]}
	}

	line.SetXAxis(xAxis)
	line.AddSeries("MSE", yAxis)

	f, _ := os.Create("learning_curve.html")
	line.Render(f)
}

// createFeatureImportancePlot creates a bar chart of feature importance
func createFeatureImportancePlot(features []string, weights []float64) {
	bar := charts.NewBar()
	bar.SetGlobalOptions(
		charts.WithTitleOpts(opts.Title{
			Title: "Feature Importance",
		}),
		charts.WithInitializationOpts(opts.Initialization{
			Theme: types.ThemeWesteros,
		}),
		charts.WithXAxisOpts(opts.XAxis{
			Name: "Features",
		}),
		charts.WithYAxisOpts(opts.YAxis{
			Name: "Weight Magnitude",
		}),
	)

	// Prepare data
	yAxis := make([]opts.BarData, len(weights))
	for i := range weights {
		yAxis[i] = opts.BarData{Value: math.Abs(weights[i])}
	}

	bar.SetXAxis(features)
	bar.AddSeries("Weight Magnitude", yAxis)

	f, _ := os.Create("feature_importance.html")
	bar.Render(f)
}

// loadData loads and preprocesses the wine dataset
func loadData(filename string) ([]Wine, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	// Skip header row
	records = records[1:]

	var wines []Wine
	for _, record := range records {
		features := make([]float64, len(record)-1)
		for i := 0; i < len(record)-1; i++ {
			features[i], err = strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, err
			}
		}

		quality, err := strconv.ParseFloat(record[len(record)-1], 64)
		if err != nil {
			return nil, err
		}

		wines = append(wines, Wine{
			features: features,
			quality:  quality,
		})
	}

	return wines, nil
}

// splitData splits the dataset into training and testing sets
func splitData(data []Wine, trainRatio float64) ([]Wine, []Wine) {
	trainSize := int(float64(len(data)) * trainRatio)
	return data[:trainSize], data[trainSize:]
}

// calculateMSE calculates the Mean Squared Error
func calculateMSE(model *LinearRegression, data []Wine) float64 {
	var totalError float64
	for _, wine := range data {
		prediction := model.predict(wine.features)
		error := prediction - wine.quality
		totalError += error * error
	}
	return totalError / float64(len(data))
}

// calculateRMSE calculates the Root Mean Squared Error
func calculateRMSE(mse float64) float64 {
	return math.Sqrt(mse)
}

// calculateR2 calculates the R-squared score
func calculateR2(model *LinearRegression, data []Wine) float64 {
	var ssTot, ssRes float64
	var meanQuality float64

	// Calculate mean quality
	for _, wine := range data {
		meanQuality += wine.quality
	}
	meanQuality /= float64(len(data))

	// Calculate SS_tot and SS_res
	for _, wine := range data {
		prediction := model.predict(wine.features)
		ssTot += math.Pow(wine.quality-meanQuality, 2)
		ssRes += math.Pow(wine.quality-prediction, 2)
	}

	return 1 - (ssRes / ssTot)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Load the data
	wines, err := loadData("wine.csv")
	if err != nil {
		fmt.Printf("Error loading data: %v\n", err)
		return
	}

	// Perform 5-fold cross-validation
	fmt.Println("\nPerforming 5-fold cross-validation...")
	cvScores := crossValidate(wines, 5, 0.0001, 1000)

	fmt.Printf("\nCross-validation R² scores:\n")
	sum := 0.0
	for i, score := range cvScores {
		fmt.Printf("Fold %d: %.4f\n", i+1, score)
		sum += score
	}
	fmt.Printf("Mean R²: %.4f\n", sum/float64(len(cvScores)))

	// Train final model on full dataset
	fmt.Println("\nTraining final model...")
	model := NewLinearRegression(len(wines[0].features))
	normalizedData := model.normalizeFeatures(wines)
	trainData, testData := splitData(normalizedData, 0.8)
	mseHistory := model.train(trainData, 0.0001, 1000)

	// Create learning curve plot
	createLearningCurvePlot(mseHistory)

	// Evaluate the model
	trainMSE := calculateMSE(model, trainData)
	testMSE := calculateMSE(model, testData)
	trainRMSE := calculateRMSE(trainMSE)
	testRMSE := calculateRMSE(testMSE)
	trainR2 := calculateR2(model, trainData)
	testR2 := calculateR2(model, testData)

	fmt.Printf("\nFinal Model Evaluation:\n")
	fmt.Printf("Training RMSE: %.4f\n", trainRMSE)
	fmt.Printf("Testing RMSE: %.4f\n", testRMSE)
	fmt.Printf("Training R²: %.4f\n", trainR2)
	fmt.Printf("Testing R²: %.4f\n", testR2)

	// Print and visualize feature importance
	features := []string{
		"Fixed Acidity",
		"Volatile Acidity",
		"Citric Acid",
		"Residual Sugar",
		"Chlorides",
		"Free Sulfur Dioxide",
		"Total Sulfur Dioxide",
		"Density",
		"pH",
		"Sulphates",
		"Alcohol",
	}

	fmt.Printf("\nFeature Weights (Normalized):\n")
	for i, weight := range model.weights {
		fmt.Printf("%s: %.4f\n", features[i], weight)
	}
	fmt.Printf("Bias: %.4f\n", model.bias)

	// Create feature importance plot
	createFeatureImportancePlot(features, model.weights)

	fmt.Printf("\nVisualization files created:\n")
	fmt.Printf("1. learning_curve.html - Shows how MSE changes during training\n")
	fmt.Printf("2. feature_importance.html - Shows the relative importance of each feature\n")
}
