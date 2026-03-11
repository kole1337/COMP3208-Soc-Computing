package recommender;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


public class RecommenderSystem {


    private Map<Integer, Map<Integer, Double>> userRatings;
    private Map<Integer, Map<Integer, Double>> itemUsers;
    private Map<Integer, Double> userMeans;
    private Map<Integer, Double> userVectorNorms;

    private double globalAverage = 0.0;

    public RecommenderSystem() {
        userRatings = new HashMap<>();
        itemUsers = new HashMap<>();
        userMeans = new HashMap<>();
        userVectorNorms = new HashMap<>();
    }

    public void train(String trainingFilePath) {
        long totalRatings = 0;
        double sumRatings = 0.0;

        try (BufferedReader br = new BufferedReader(new FileReader(trainingFilePath))) {
            String line;

            while ((line = br.readLine()) != null) {
                if(line.toLowerCase().contains("user")) continue;

                String[] parts = line.split(",");
                if (parts.length < 3) continue;

                int userId = Integer.parseInt(parts[0].trim());
                int itemId = Integer.parseInt(parts[1].trim());
                double rating = Double.parseDouble(parts[2].trim());

                userRatings.putIfAbsent(userId, new HashMap<>());
                userRatings.get(userId).put(itemId, rating);

                itemUsers.putIfAbsent(itemId, new HashMap<>());
                itemUsers.get(itemId).put(userId, rating);

                sumRatings += rating;
                totalRatings++;
            }
        } catch (IOException e) {
            System.err.println("Error reading training file: " + e.getMessage());
        }

        if (totalRatings > 0) {
            globalAverage = sumRatings / totalRatings;
        }

        for (Map.Entry<Integer, Map<Integer, Double>> entry : userRatings.entrySet()) {
            int userId = entry.getKey();
            Map<Integer, Double> ratings = entry.getValue();

            double sum = 0.0;
            double sumSquares = 0.0;

            for (double r : ratings.values()) {
                sum += r;
                sumSquares += (r * r);
            }

            userMeans.put(userId, sum / ratings.size());
            userVectorNorms.put(userId, Math.sqrt(sumSquares));
        }
        System.out.println("Training complete. Loaded " + userRatings.size() + " users.");
    }

    private double calculateCosineSimilarity(int userA, int userB) {
        Map<Integer, Double> ratingsA = userRatings.get(userA);
        Map<Integer, Double> ratingsB = userRatings.get(userB);

        if (ratingsA == null || ratingsB == null) return 0.0;

        double dotProduct = 0.0;
        for (Map.Entry<Integer, Double> entry : ratingsA.entrySet()) {
            int itemId = entry.getKey();
            if (ratingsB.containsKey(itemId)) {
                dotProduct += (entry.getValue() * ratingsB.get(itemId));
            }
        }

        if (dotProduct == 0.0) return 0.0;

        double normA = userVectorNorms.get(userA);
        double normB = userVectorNorms.get(userB);

        return dotProduct / (normA * normB);
    }

    public double predictRating(int targetUser, int targetItem) {
        if (!userRatings.containsKey(targetUser)) {
            return globalAverage;
        }

        double targetUserMean = userMeans.get(targetUser);

        if (!itemUsers.containsKey(targetItem)) {
            return targetUserMean;
        }

        double numerator = 0.0;
        double denominator = 0.0;

        Map<Integer, Double> neighbors = itemUsers.get(targetItem);

        for (Map.Entry<Integer, Double> neighbor : neighbors.entrySet()) {
            int neighborId = neighbor.getKey();

            if (neighborId == targetUser) continue;

            double similarity = calculateCosineSimilarity(targetUser, neighborId);

            if (similarity > 0) {
                double neighborRating = neighbor.getValue();
                double neighborMean = userMeans.get(neighborId);

                numerator += similarity * (neighborRating - neighborMean);
                denominator += Math.abs(similarity);
            }
        }

        if (denominator == 0.0) {
            return targetUserMean;
        }

        double prediction = targetUserMean + (numerator / denominator);

        if (prediction > 5.0) return 5.0;
        if (prediction < 0.5) return 0.5;

        return prediction;
    }

    public void generatePredictions(String testFilePath, String outputFilePath) {
        try (BufferedReader br = new BufferedReader(new FileReader(testFilePath));
             BufferedWriter bw = new BufferedWriter(new FileWriter(outputFilePath))) {

            String line;

            while ((line = br.readLine()) != null) {
                if(line.toLowerCase().contains("user")) continue;

                String[] parts = line.split(",");
                if (parts.length < 3) continue;

                int userId = Integer.parseInt(parts[0].trim());
                int itemId = Integer.parseInt(parts[1].trim());
                long timestamp = Long.parseLong(parts[2].trim());

                double predictedRating = predictRating(userId, itemId);

                bw.write(userId + "," + itemId + "," + String.format("%.4f", predictedRating) + "," + timestamp + "\n");
            }
            System.out.println("Predictions successfully generated to " + outputFilePath);

        } catch (IOException e) {
            System.err.println("Error processing test file: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        RecommenderSystem recommender = new RecommenderSystem();
        
        String trainingFile = "recommender/csv/train_100k_withratings.csv";
        String testFile = "recommender/csv/test_100k_withoutratings.csv";
        String outputFile = "results.csv";
        
        System.out.println("Starting Recommender System Training...");
        recommender.train(trainingFile);
        
        System.out.println("Generating Predictions...");
        recommender.generatePredictions(testFile, outputFile);
    }
}
