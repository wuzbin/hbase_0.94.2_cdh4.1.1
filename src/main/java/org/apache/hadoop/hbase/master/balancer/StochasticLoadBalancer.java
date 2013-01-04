/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.hadoop.hbase.master.balancer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.lang.mutable.MutableInt;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.ClusterStatus;
import org.apache.hadoop.hbase.HRegionInfo;
import org.apache.hadoop.hbase.ServerName;
import org.apache.hadoop.hbase.master.MasterServices;
import org.apache.hadoop.hbase.master.RegionPlan;
import org.apache.hadoop.hbase.regionserver.StoreFile;

/**
 * This is a best effort load balancer. Given a Cost function F(C) => x It will
 * randomly try and mutate the cluster to Cprime. If F(Cprime) < F(C) then the
 * new cluster state becomes the plan.
 *
 * This balancer is best used with hbase.master.loadbalance.bytable set to false
 * so that the balancer gets the full region load picture.
 */
public class StochasticLoadBalancer extends BaseLoadBalancer {

  private static final Random RANDOM = new Random(System.currentTimeMillis());
  private static final Log LOG = LogFactory.getLog(StochasticLoadBalancer.class);
  private final RegionLocationFinder regionFinder = new RegionLocationFinder();

  // values are
  private int maxSteps = 20000;
  private int stepsPerRegion = 100;
  private float loadMultiplier = 15;
  private float tableMultiplier = 5;
  private float localityMultiplier = 5;


  @Override
  public void setConf(Configuration conf) {
    super.setConf(conf);
    regionFinder.setConf(conf);

    maxSteps = conf.getInt("hbase.master.balancer.stochastic.maxSteps", 20000);
    stepsPerRegion = conf.getInt("hbase.master.balancer.stochastic.stepsPerRegion", 100);
    loadMultiplier = conf.getFloat("hbase.master.balancer.stochastic.regionLoadCost", 15f);
    tableMultiplier = conf.getFloat("hbase.master.balancer.stochastic.tableLoadCost", 5f);
    localityMultiplier = conf.getFloat("hbase.master.balancer.stochastic.localityCost", 5f);

  }

  @Override
  public void setClusterStatus(ClusterStatus st) {
    regionFinder.setClusterStatus(st);
  }

  @Override
  public void setMasterServices(MasterServices masterServices) {
    this.services = masterServices;
    this.regionFinder.setServices(masterServices);
  }

  /**
   * Given the cluster state this will try and approach an optimal balance. This
   * should always approach the optimal state given enough steps.
   */
  @Override
  public List<RegionPlan> balanceCluster(Map<ServerName, List<HRegionInfo>> clusterState) {

    // No need to balance a one node cluster.
    if (clusterState.size() <= 1) {
      LOG.debug("Skipping load balance as cluster has only one node.");
      return null;
    }

    long startTime = System.currentTimeMillis();

    // Keep track of servers to iterate through them.
    List<ServerName> servers = new ArrayList<ServerName>(clusterState.keySet());
    Map<HRegionInfo, ServerName> initialRegionMapping = createRegionMapping(clusterState);
    double currentCost, newCost, initCost;
    currentCost = newCost = initCost= computeCost(initialRegionMapping, clusterState);

    int computedMaxSteps =
        Math.min(this.maxSteps, (initialRegionMapping.size() * this.stepsPerRegion));
    // Perform a stochastic walk to see if we can get a good fit.
    for (int step = 0; step < computedMaxSteps; step++) {

      // try and perform a mutation
      for (ServerName leftServer : servers) {

        // What server are we going to be swapping regions with ?
        ServerName rightServer = pickOtherServer(leftServer, servers);
        if (rightServer == null) {
          continue;
        }

        // Get the regions.
        List<HRegionInfo> leftRegionList = clusterState.get(leftServer);
        List<HRegionInfo> rightRegionList = clusterState.get(rightServer);

        // Pick what regions to swap around.
        // If we get a null for one then this isn't a swap just a move
        HRegionInfo lRegion = pickRandomRegion(leftRegionList, 0);
        HRegionInfo rRegion = pickRandomRegion(rightRegionList, 0.5);

        // We randomly picked to do nothing.
        if (lRegion == null && rRegion == null) {
          continue;
        }

        if (rRegion != null) {
          leftRegionList.add(rRegion);
        }

        if (lRegion != null) {
          rightRegionList.add(lRegion);
        }

        newCost = computeCost(initialRegionMapping, clusterState);

        // Should this be kept?
        if (newCost < currentCost)  {
          currentCost = newCost;
        } else {
          // Put things back the way they were before.
          if (rRegion != null) {
            leftRegionList.remove(rRegion);
            rightRegionList.add(rRegion);
          }

          if (lRegion != null) {
            rightRegionList.remove(lRegion);
            leftRegionList.add(lRegion);
          }
        }
      }

    }

    long endTime = System.currentTimeMillis();


    if (initCost > currentCost) {
      List<RegionPlan> plans = createRegionPlans(initialRegionMapping, clusterState);

      LOG.debug("Finished computing new laod balance plan.  Computation took "
          + (endTime - startTime) + "ms to try " + computedMaxSteps
          + " different iterations.  Found a solution that moves " + plans.size()
          + " regions; Going from a computed cost of " + initCost + " to a new cost of "
          + currentCost);
      return plans;
    }
    LOG.debug("Could not find a better load balance plan.  Tried " + computedMaxSteps
        + " different configurations in " + (endTime - startTime)
        + "ms, and did not find anything with a computed cost less than " + initCost);
    return null;
  }

  /**
   * From a cluster state compute the plan to get there.
   */
  private List<RegionPlan> createRegionPlans(Map<HRegionInfo, ServerName> initialRegionMapping,
      Map<ServerName, List<HRegionInfo>> clusterState) {
    List<RegionPlan> plans = new LinkedList<RegionPlan>();

    for (Entry<ServerName, List<HRegionInfo>> entry : clusterState.entrySet()) {
      ServerName newServer = entry.getKey();

      for (HRegionInfo region : entry.getValue()) {
        ServerName initialServer = initialRegionMapping.get(region);
        if (!newServer.equals(initialServer)) {
          LOG.trace("Moving Region " + region.getEncodedName() + " from server "
              + initialServer.getHostname() + " to " + newServer.getHostname());
          RegionPlan rp = new RegionPlan(region, initialServer, newServer);
          plans.add(rp);
        }
      }
    }
    return plans;
  }

  /**
   * Create a map that will represent the initial location of regions on a
   * {@link ServerName}
   * @param clusterState
   * @return A map of {@link HRegionInfo} to the {@link ServerName} that is
   *         currently hosting that region
   */
  private Map<HRegionInfo, ServerName> createRegionMapping(
      Map<ServerName, List<HRegionInfo>> clusterState) {
    Map<HRegionInfo, ServerName> mapping = new HashMap<HRegionInfo, ServerName>();

    for (Entry<ServerName, List<HRegionInfo>> entry : clusterState.entrySet()) {
      for (HRegionInfo region : entry.getValue()) {
        mapping.put(region, entry.getKey());
      }
    }
    return mapping;
  }

  /**
   * From a list of regions pick a random one. Null can be returned which
   * {@link StochasticLoadBalancer#balanceCluster(Map)} use to try a region move
   * rather than swap.
   * @param regions list of regions.
   * @param chanceOfNoSwap Chance that this will decide to try a move rather
   *          than a swap.
   * @return a random {@link HRegionInfo} or null if a non-symetrical move is suggested.
   */
  private HRegionInfo pickRandomRegion(List<HRegionInfo> regions, double chanceOfNoSwap) {

    if (regions.isEmpty() || RANDOM.nextFloat() < chanceOfNoSwap) {
      return null;
    }

    int index = RANDOM.nextInt(regions.size());

    // Don't move meta regions.
    if (regions.get(index).isMetaRegion() == false) {
      return regions.remove(index);
    }

    return null;
  }

  /**
   * Given a server we will want to switch regions with another server. This
   * function picks a random server from the list.
   * @param server Current Server. This server will never be the return value.
   * @param allServers list of all server from which to pick
   * @return random server.  Null if no other servers were found.
   */
  private ServerName pickOtherServer(ServerName server, List<ServerName> allServers) {
    ServerName s = null;
    int count = 0;
    while (count < 100 && (s == null || s.equals(server))) {
      count++;
      s = allServers.get(RANDOM.nextInt(allServers.size()));
    }

    // If nothing but the current server was found return null.
    return s.equals(server) ? null : s;
  }

  protected double computeCost(Map<HRegionInfo, ServerName> initialRegionMapping,
      Map<ServerName, List<HRegionInfo>> clusterState) {

    double moveCost = computeMoveCost(initialRegionMapping, clusterState);

    // TODO: add a per region num requests/sec cost.
    double regionCountSkewCost = loadMultiplier * computeSkewLoadCost(clusterState);
    double tableSkewCost = tableMultiplier * computeTableSkewLoadCost(clusterState);
    double localityCost =
        localityMultiplier * computeDataLocalityCost(initialRegionMapping, clusterState);

    double total = moveCost + regionCountSkewCost + tableSkewCost + localityCost;
    LOG.trace("Computed weights for a potential balancing total = " + total + " moveCost = "
        + moveCost + " regionCountSkewCost = " + regionCountSkewCost + " tableSkewCost = "
        + tableSkewCost + " localityCost = " + localityCost);
    return total;
  }

  /**
   * Given the starting state of the regions and a potential ending state
   * compute cost based upon the number of regions that have moved.
   * @param initialRegionMapping The starting location of regions.
   * @param clusterState The potential new cluster state.
   * @return The cost. Between 0 and 1.
   */
  double computeMoveCost(Map<HRegionInfo, ServerName> initialRegionMapping,
      Map<ServerName, List<HRegionInfo>> clusterState) {
    float moveCost = 0;
    for (Entry<ServerName, List<HRegionInfo>> entry : clusterState.entrySet()) {
      for (HRegionInfo region : entry.getValue()) {
        if (initialRegionMapping.get(region) != entry.getKey()) {
          moveCost += 1;
        }
      }
    }
    return scale(0, initialRegionMapping.size(), moveCost);
  }

  /**
   * Compute the cost of a potential cluster state from skew in number of
   * regions on a cluster
   * @param clusterState The proposed cluster state
   * @return The cost of region load imbalance.
   */
  double computeSkewLoadCost(Map<ServerName, List<HRegionInfo>> clusterState) {
    double skewCost = 0;
    double numRegions = 0;
    DescriptiveStatistics stats = new DescriptiveStatistics();
    for (List<HRegionInfo> regions : clusterState.values()) {
      int size = regions.size();
      numRegions += regions.size();
      stats.addValue(size);
    }

    double mean = stats.getMean();
    double max = (mean * (clusterState.size() - 1)) + (Math.abs(mean - numRegions));

    for (List<HRegionInfo> regions : clusterState.values()) {
      skewCost += Math.abs(mean - regions.size());

    }

    return scale(0, max, skewCost);
  }

  /**
   * Compute the cost of a potential cluster configuration based upon how evenly
   * distributed tables are.
   * @param clusterState Proposed cluster state.
   * @return Cost of imbalance in table.
   */
  double computeTableSkewLoadCost(Map<ServerName, List<HRegionInfo>> clusterState) {

    Map<String, MutableInt> tableRegionsTotal = new HashMap<String, MutableInt>();
    Map<String, MutableInt> tableRegionsOnCurrentServer = new HashMap<String, MutableInt>();
    Map<String, Integer> tableCostSeenSoFar = new HashMap<String, Integer>();
    // Go through everything per server
    for (Entry<ServerName, List<HRegionInfo>> entry : clusterState.entrySet()) {
      tableRegionsOnCurrentServer.clear();

      // For all of the regions count how many are from each table
      for (HRegionInfo region : entry.getValue()) {
        String tableName = region.getTableNameAsString();

        // See if this table already has a count on this server
        MutableInt regionsOnServerCount = tableRegionsOnCurrentServer.get(tableName);

        // If this is the first time we've seen this table on this server
        // create a new mutable int.
        if (regionsOnServerCount == null) {
          regionsOnServerCount = new MutableInt(0);
          tableRegionsOnCurrentServer.put(tableName, regionsOnServerCount);
        }

        // Increment the count of how many regions from this table are host on
        // this server
        regionsOnServerCount.increment();

        // Now count the number of regions in this table.
        MutableInt totalCount = tableRegionsTotal.get(tableName);

        // If this is the first region from this table create a new counter for
        // this table.
        if (totalCount == null) {
          totalCount = new MutableInt(0);
          tableRegionsTotal.put(tableName, totalCount);
        }
        totalCount.increment();
      }

      // Now go through all of the tables we have seen and keep the max number
      // of regions of this table a single region server is hosting.
      for (String tableName : tableRegionsOnCurrentServer.keySet()) {
        Integer thisCount = tableRegionsOnCurrentServer.get(tableName).toInteger();
        Integer maxCountSoFar = tableCostSeenSoFar.get(tableName);

        if (maxCountSoFar == null || thisCount.compareTo(maxCountSoFar) > 0) {
          tableCostSeenSoFar.put(tableName, thisCount);
        }

      }
    }

    double max = 0;
    double min = 0;
    double value = 0;

    // Comput the min, value, and max.
    for (String tableName : tableRegionsTotal.keySet()) {
      max += tableRegionsTotal.get(tableName).doubleValue();
      min += tableRegionsTotal.get(tableName).doubleValue() / (double) clusterState.size();
      value += tableCostSeenSoFar.get(tableName).doubleValue();

    }
    return scale(min, max, value);
  }

  /**
   * Compute a cost of a potential cluster configuration based upon where
   * {@link StoreFile}s are located.
   * @param clusterState The state of the cluster
   * @return A cost between 0 and 1. 0 Means all regions are on the sever with
   *         the most local store files.
   */
  double computeDataLocalityCost(Map<HRegionInfo, ServerName> initialRegionMapping,
      Map<ServerName, List<HRegionInfo>> clusterState) {

    double max = 0;
    double cost = 0;

    // If there's no master so there's no way anything else works.
    if (this.services == null) return cost;

    for (Entry<ServerName, List<HRegionInfo>> entry : clusterState.entrySet()) {
      ServerName sn = entry.getKey();
      for (HRegionInfo region : entry.getValue()) {

        max += 1;

        // Only compute the data locality for moved regions.
        if (initialRegionMapping.equals(sn)) {
          continue;
        }

        List<ServerName> dataOnServers = regionFinder.getTopBlockLocations(region);

        // If we can't find where the data is getTopBlock returns null.
        // so count that as being the best possible.
        if (dataOnServers == null) {
          continue;
        }

        int index = dataOnServers.indexOf(sn);
        if (index < 0) {
          cost += 1;
        } else {
          cost += (double) index / (double) dataOnServers.size();
        }

      }
    }
    return scale(0, max, cost);
  }

  /**
   * Scale the value between 0 and 1.
   * @param min Min value
   * @param max The Max value
   * @param value The value to be scaled.
   * @return The scaled value.
   */
  private double scale(double min, double max, double value) {
    if (max == 0 || value == 0) {
      return 0;
    }

    return Math.max(0d, Math.min(1d, (value - min) / max));
  }
}
