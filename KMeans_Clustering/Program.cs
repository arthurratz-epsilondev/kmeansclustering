using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Globalization;

namespace KMeansClustering
{
    class Program
    {
        static List<Tuple<List<double>, string>> LoadDataFromFile(string filename)
        {
            // Instantiate a list of items
            List<Tuple<List<double>, string>> Items =
                new List<Tuple<List<double>, string>>();

            // Open .csv-file for reading
            using (System.IO.StreamReader file =
                new System.IO.StreamReader(filename))
            {
                string buf = "\0";
                // Read .csv-file line-by-line
                while ((buf = file.ReadLine()) != null)
                {
                    List<double> features = new List<double>();
                    // Split each line into a list of values
                    foreach (string param in buf.Split(',').ToArray())
                        // For each value perform a check if it's a floating-point value
                        if (Regex.Match(param, @"[.]", RegexOptions.IgnoreCase).Success)
                            // If so, add the current value to the list of item's features
                            features.Add(Double.Parse(param, CultureInfo.InvariantCulture));

                    // Add each pair of a vector of features and its label to the list of items
                    Items.Add(new Tuple<List<double>, string>(features,
                         buf.Split(',').ElementAt(buf.Split(',').Length - 1)));
                }

                // Close the file
                file.Close();
            }

            // Return the list of items
            return Items;
        }
        static double EuclDist(List<double> params1,
            List<double> params2, bool squared = false)
        {
            double distance = .0f;
            // For each dimension, compute the squared difference
            // between two features and accumulate the result in 
            // the distance variable
            for (int Index = 0; Index < params1.Count(); Index++)
                distance += Math.Pow((params1[Index] - params2[Index]), 2);

            // Return a squared or regular distance value
            return !squared ? Math.Sqrt(distance) : distance;
        }
        static int Euclidean_Step(List<Tuple<List<double>,
            string>> items, int centroid)
        {
            List<Tuple<int, double>> dists = new List<Tuple<int, double>>();
            // For each item compute the distance two an already existing centroid
            for (int Index = 0; Index < items.Count(); Index++)
                // Add each distance to the list of distances
                dists.Add(new Tuple<int, double>(Index,
                    EuclDist(items[centroid].Item1, items[Index].Item1, true)));

            // Find an item with the maxima distance to the centroid specified
            return dists.Find(obj => obj.Item2 ==
                dists.Max(dist => dist.Item2)).Item1;
        }
        static List<Tuple<int, List<int>>> Lloyd_Step(
            List<Tuple<List<double>, string>> items,
            List<int> centroids, List<List<double>> c_mass = null)
        {
            List<Tuple<int, List<int>>> clusters =
                new List<Tuple<int, List<int>>>();

            // Pre-build a set of new clusters based on the centroids index values
            if (centroids != null)
            {
                for (int Index = 0; Index < centroids.Count(); Index++)
                    clusters.Add(new Tuple<int, List<int>>(
                        centroids[Index], new List<int>()));
            }

            else
            {
                for (int Index = 0; Index < c_mass.Count(); Index++)
                    clusters.Add(new Tuple<int, List<int>>(Index, new List<int>()));
            }

            for (int Index = 0; Index < items.Count(); Index++)
            {
                double dist_min = .0f; int cluster = -1;

                if (centroids != null)
                {
                    // For each item compute the distance to each centroid
                    // finding an item with the smallest distance to a current centroid
                    for (int cnt = 0; cnt < centroids.Count(); cnt++)
                    {
                        double distance = EuclDist(items[Index].Item1,
                            items[centroids[cnt]].Item1, true);

                        if ((distance <= dist_min) || (cluster == -1))
                        {
                            dist_min = distance; cluster = cnt;
                        }
                    }

                    // Assign the following item to a cluster with centroid
                    // having the smallest distance
                    var cluster_target = clusters.Find(
                        obj => obj.Item1 == centroids[cluster]);

                    if (cluster_target != null)
                        cluster_target.Item2.Add(Index);
                }

                else
                {
                    for (int cnt = 0; cnt < c_mass.Count(); cnt++)
                    {
                        // For each item compute the distance to each centroid
                        // finding an item with the smallest distance to a current centroid
                        double distance = EuclDist(items[Index].Item1, c_mass[cnt], true);
                        if ((distance <= dist_min) || (cluster == -1))
                        {
                            dist_min = distance; cluster = cnt;
                        }
                    }

                    // Assign the following item to a cluster with centroid
                    // having the smallest distance
                    var cluster_target = clusters.Find(
                        obj => obj.Item1 == cluster);

                    if (cluster_target != null)
                        cluster_target.Item2.Add(Index);
                }
            }

            // Return a list of clusters
            return clusters;
        }
        static Tuple<List<int>, List<Tuple<int, List<int>>>> KmeansPlusPlus(
            List<Tuple<List<double>, string>> items, int k)
        {
            // Initialize a list of centroids with a single centroid
            // randomly selected
            List<int> centroids = new List<int>() {
                new Random().Next(items.Count())
            };

            if (centroids.Count() == 1)
                // Compute the second centroid by invoking the Euclidean_Step(...)
                // function and append it to the list of centroids
                centroids.Add(Euclidean_Step(items, centroids[0]));

            List<Tuple<int, List<int>>> targets = null;
            // Compute the other initial centroids
            for (int count = k - 2; count >= 0; count--)
            {
                // Perform a Lloyd's step to obtain a list of initial clusters
                List<Tuple<int, List<int>>> clusters =
                    Lloyd_Step(items, centroids);

                double dist_max = .0f; int cmax_index = -1, dmax_index = -1;
                // For each cluster compute an item with the largest distance
                // to its centroid
                for (int Index = 0; Index < clusters.Count(); Index++)
                {
                    int centroid = clusters[Index].Item1;
                    for (int pt = 0; pt < clusters[Index].Item2.Count(); pt++)
                    {
                        double distance = EuclDist(items[centroid].Item1,
                            items[clusters[Index].Item2[pt]].Item1);

                        if ((distance > dist_max) ||
                            (cmax_index == -1) || (dmax_index == -1))
                        {
                            dist_max = distance;
                            cmax_index = Index; dmax_index = pt;
                        }
                    }
                }

                if (count > 0)
                    // Add the following item index to the list of centroids
                    centroids.Add(clusters[cmax_index].Item2[dmax_index]);

                targets = (clusters.Count() > 0) && (count == 0) ? clusters : null;
            }

            // Return a tuple of a list of centroids and a list of pre-built clusters
            return new Tuple<List<int>, List<Tuple<int, List<int>>>>(centroids, targets);
        }
        static List<List<double>> RecomputeCentroids(
            List<Tuple<int, List<int>>> clusters,
            List<Tuple<List<double>, string>> Items)
        {
            List<List<double>> centroids_new =
                new List<List<double>>();

            // For each cluster re-compute the new centroids
            for (int clu = 0; clu < clusters.Count(); clu++)
            {
                List<int> c_items = clusters[clu].Item2;
                List<double> centroid = new List<double>();
                // For a list of items, compute the average of each coordinate
                for (int i = 0; i < Items[c_items[0]].Item1.Count(); i++)
                {
                    if (Items[c_items[0]].Item1.Count() > 0)
                    {
                        double avg = .0f;
                        for (int Index = 0; Index < c_items.Count(); Index++)
                            avg += Items[c_items[Index]].Item1[i] / c_items.Count();

                        centroid.Add(avg);
                    }
                }

                // Add a new centroid to the list of centroids
                centroids_new.Add(centroid);
            }

            // Return a list of new centroids
            return centroids_new;
        }
        static List<Tuple<int, List<int>>> KMeans(
            List<Tuple<List<double>, string>> Items, int k)
        {
            // Find k - initial centroids using k-means++ procedure
            Tuple<List<int>, List<Tuple<int,
                List<int>>>> result = KmeansPlusPlus(Items, k);

            // Instantiate the list of target clusters
            List<Tuple<int, List<int>>> clusters_target =
                new List<Tuple<int, List<int>>>();

            List<int> centroids = result.Item1;
            List<Tuple<int, List<int>>> clusters = result.Item2;

            while (clusters.Count() > 1)
            {
                // Re-compute the centroids of the pre-built clusters
                List<List<double>> centroids_new =
                    RecomputeCentroids(clusters, Items);

                // Perform a Lloyd step to partition the dataset
                List<Tuple<int, List<int>>> clusters_new =
                    Lloyd_Step(Items, null, centroids_new);

                // Perform a check if we're not producing the same clusters
                // and the k-means procedure has not converged.
                // If not, proceed with the next clustering phase
                for (int clu = 0; clu < clusters.Count(); clu++)
                {
                    if ((Compare(clusters[clu].Item2, clusters_new[clu].Item2)))
                    {
                        clusters_target.Add(clusters[clu]);
                        clusters.RemoveAt(clu); clusters_new.RemoveAt(clu); clu--;
                    }
                }

                if (clusters_new.Count() > 1)
                    clusters = clusters_new;
            }

            return clusters_target;
        }
        static bool Compare(List<int> list1, List<int> list2)
        {
            return list2.Intersect(list1).Count() == list2.Count();
        }
        static void Main(string[] args)
        {
            Console.WriteLine("K-Means Clustering Algorithm v.1.0 by Arthur V. Ratz @ CodeProject.Com");
            Console.WriteLine("======================================================================");

            string filename = "\0";
            Console.Write("Enter a filename: ");
            filename = Console.ReadLine();

            int k = 0;
            Console.Write("Enter a number of clusters: ");
            k = Int32.Parse(Console.ReadLine()); Console.WriteLine();

            List<Tuple<List<double>, string>> Items =
                LoadDataFromFile(filename);

            List<Tuple<int, List<int>>> clusters_target =
                   KMeans(Items, k);

            for (int clu = 0; clu < clusters_target.Count(); clu++)
            {
                Console.WriteLine("Cluster = {0}", clu + 1);
                for (int Index = 0; Index < clusters_target[clu].Item2.Count(); Index++)
                {
                    for (int Item = 0; Item < Items[clusters_target[clu].Item2[Index]].Item1.Count(); Item++)
                        Console.Write("{0} ", Items[clusters_target[clu].Item2[Index]].Item1[Item]);

                    Console.WriteLine("{0}", Items[clusters_target[clu].Item2[Index]].Item2);
                }

                Console.WriteLine("\n");
            }

            Console.ReadKey();
        }
    }
}

