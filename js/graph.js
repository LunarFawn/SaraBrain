/**
 * graph.js — D3.js force-directed graph visualization for Sara Brain.
 */

const NODE_COLORS = {
  concept: "#a8d8ea",
  property: "#ffcfdf",
  relation: "#fefdca",
  association: "#c4b5fd",
};

const NODE_RADIUS = {
  concept: 24,
  property: 20,
  relation: 16,
  association: 22,
};

let svg, simulation, linkGroup, nodeGroup, labelGroup;
let currentNodes = [];
let currentLinks = [];
let width, height;

/**
 * Initialize the D3 graph in the given container element.
 */
export function initGraph(containerId) {
  const container = document.getElementById(containerId);
  const rect = container.getBoundingClientRect();
  width = rect.width;
  height = rect.height;

  svg = d3
    .select(`#${containerId}`)
    .append("svg")
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("viewBox", `0 0 ${width} ${height}`);

  // Arrow marker for directed edges
  svg
    .append("defs")
    .append("marker")
    .attr("id", "arrowhead")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 28)
    .attr("refY", 0)
    .attr("markerWidth", 8)
    .attr("markerHeight", 8)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", "#666");

  const zoomGroup = svg.append("g").attr("class", "zoom-layer");

  linkGroup = zoomGroup.append("g").attr("class", "links");
  nodeGroup = zoomGroup.append("g").attr("class", "nodes");
  labelGroup = zoomGroup.append("g").attr("class", "labels");

  const zoom = d3.zoom()
    .scaleExtent([0.1, 4])
    .on("zoom", (event) => {
      zoomGroup.attr("transform", event.transform);
    });

  svg.call(zoom);

  simulation = d3
    .forceSimulation()
    .force(
      "link",
      d3
        .forceLink()
        .id((d) => d.id)
        .distance(120)
    )
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(35))
    .on("tick", ticked);

  // Handle window resize
  const resizeObserver = new ResizeObserver(() => {
    const newRect = container.getBoundingClientRect();
    width = newRect.width;
    height = newRect.height;
    svg.attr("viewBox", `0 0 ${width} ${height}`);
    simulation.force("center", d3.forceCenter(width / 2, height / 2));
    simulation.alpha(0.3).restart();
  });
  resizeObserver.observe(container);
}

/**
 * Update the graph with new data. Preserves positions of existing nodes.
 */
export function updateGraph(data, onNodeClick) {
  const { nodes, links } = data;

  // Preserve positions of existing nodes
  const posMap = {};
  currentNodes.forEach((n) => {
    posMap[n.id] = { x: n.x, y: n.y, vx: n.vx, vy: n.vy };
  });

  nodes.forEach((n) => {
    if (posMap[n.id]) {
      Object.assign(n, posMap[n.id]);
    }
  });

  currentNodes = nodes;
  currentLinks = links;

  // Links
  const link = linkGroup.selectAll("line").data(links, (d) => `${d.source.id || d.source}-${d.target.id || d.target}`);
  link.exit().remove();
  const linkEnter = link
    .enter()
    .append("line")
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.6)
    .attr("marker-end", "url(#arrowhead)");
  const linkMerge = linkEnter.merge(link);
  linkMerge.attr("stroke-width", (d) => Math.max(1, Math.min(5, d.strength * 1.5)));

  // Nodes
  const node = nodeGroup.selectAll("circle").data(nodes, (d) => d.id);
  node.exit().remove();
  const nodeEnter = node
    .enter()
    .append("circle")
    .attr("r", (d) => NODE_RADIUS[d.type] || 18)
    .attr("fill", (d) => NODE_COLORS[d.type] || "#ccc")
    .attr("stroke", "#fff")
    .attr("stroke-width", 2)
    .attr("cursor", "pointer")
    .call(drag(simulation));
  const nodeMerge = nodeEnter.merge(node);

  // Click handler
  nodeMerge.on("click", (event, d) => {
    if (onNodeClick) onNodeClick(d);
  });

  // Tooltip
  nodeMerge.select("title").remove();
  nodeMerge.append("title").text((d) => `${d.label} (${d.type})`);

  // Labels
  const label = labelGroup.selectAll("text").data(nodes, (d) => d.id);
  label.exit().remove();
  const labelEnter = label
    .enter()
    .append("text")
    .attr("text-anchor", "middle")
    .attr("dy", ".35em")
    .attr("font-size", "11px")
    .attr("font-family", "'JetBrains Mono', monospace")
    .attr("fill", "#333")
    .attr("pointer-events", "none");
  const labelMerge = labelEnter.merge(label);
  labelMerge.text((d) => d.label);

  // Update simulation
  simulation.nodes(nodes);
  simulation.force("link").links(links);
  simulation.alpha(0.5).restart();
}

/**
 * Animate wavefront propagation during recognition.
 * paths is an array of arrays of node IDs representing paths.
 */
export async function animateWavefront(paths) {
  // Reset all highlights
  nodeGroup.selectAll("circle").attr("stroke", "#fff").attr("stroke-width", 2);
  linkGroup.selectAll("line").attr("stroke", "#999").attr("stroke-opacity", 0.6);

  const STEP_DELAY = 400;
  const PULSE_COLORS = ["#ff6b6b", "#ffa502", "#2ed573", "#1e90ff", "#a855f7"];

  for (let pi = 0; pi < paths.length; pi++) {
    const path = paths[pi];
    const color = PULSE_COLORS[pi % PULSE_COLORS.length];

    for (let i = 0; i < path.length; i++) {
      const nodeId = path[i];

      // Highlight current node
      nodeGroup
        .selectAll("circle")
        .filter((d) => d.id === nodeId)
        .transition()
        .duration(200)
        .attr("stroke", color)
        .attr("stroke-width", 4);

      // Highlight edge to this node (if not first)
      if (i > 0) {
        const prevId = path[i - 1];
        linkGroup
          .selectAll("line")
          .filter(
            (d) =>
              (d.source.id === prevId && d.target.id === nodeId) ||
              (d.source === prevId && d.target === nodeId)
          )
          .transition()
          .duration(200)
          .attr("stroke", color)
          .attr("stroke-opacity", 1);
      }

      await new Promise((r) => setTimeout(r, STEP_DELAY));
    }
  }

  // Find intersection nodes (appear in 2+ paths)
  const visited = {};
  paths.forEach((path) => {
    path.forEach((nodeId) => {
      visited[nodeId] = (visited[nodeId] || 0) + 1;
    });
  });

  const intersections = Object.entries(visited)
    .filter(([, count]) => count >= 2)
    .map(([id]) => parseInt(id));

  // Pulse intersection nodes
  if (intersections.length > 0) {
    for (let pulse = 0; pulse < 3; pulse++) {
      nodeGroup
        .selectAll("circle")
        .filter((d) => intersections.includes(d.id))
        .transition()
        .duration(300)
        .attr("r", (d) => (NODE_RADIUS[d.type] || 18) + 8)
        .attr("stroke", "#ff0")
        .attr("stroke-width", 5)
        .transition()
        .duration(300)
        .attr("r", (d) => NODE_RADIUS[d.type] || 18)
        .attr("stroke-width", 3);

      await new Promise((r) => setTimeout(r, 600));
    }
  }

  // Fade back after 2 seconds
  await new Promise((r) => setTimeout(r, 2000));
  nodeGroup.selectAll("circle").transition().duration(500).attr("stroke", "#fff").attr("stroke-width", 2);
  linkGroup.selectAll("line").transition().duration(500).attr("stroke", "#999").attr("stroke-opacity", 0.6);
}

function ticked() {
  linkGroup
    .selectAll("line")
    .attr("x1", (d) => d.source.x)
    .attr("y1", (d) => d.source.y)
    .attr("x2", (d) => d.target.x)
    .attr("y2", (d) => d.target.y);

  nodeGroup
    .selectAll("circle")
    .attr("cx", (d) => d.x)
    .attr("cy", (d) => d.y);

  labelGroup
    .selectAll("text")
    .attr("x", (d) => d.x)
    .attr("y", (d) => d.y);
}

function drag(simulation) {
  function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }

  return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
}
