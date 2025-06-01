import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

function GraphVisualization({ graphData }) {
  const svgRef = useRef();

  useEffect(() => {
    if (!graphData) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 600;
    const height = 400;

    const simulation = d3
      .forceSimulation(graphData.nodes)
      .force('link', d3.forceLink(graphData.links).id((d) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-100))
      .force('center', d3.forceCenter(width / 2, height / 2));

    const link = svg
      .append('g')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(graphData.links)
      .enter()
      .append('line')
      .attr('stroke-width', (d) => (d.strength === 'strong' ? 2 : d.strength === 'medium' ? 1.5 : 1));

    const node = svg
      .append('g')
      .selectAll('circle')
      .data(graphData.nodes)
      .enter()
      .append('circle')
      .attr('r', (d) => graphData.nodeTypes[d.type].radius)
      .attr('fill', (d) => {
        if (d.type === 'condition' && d.probability) {
          return d.probability >= 0.6 ? '#34a853' : d.probability >= 0.3 ? '#fbbc04' : '#b81c0f';
        }
        return graphData.nodeTypes[d.type].color;
      })
      .call(d3.drag()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }));

    node.append('title').text((d) => `${d.name}${d.probability ? ` (${(d.probability * 100).toFixed(1)}%)` : ''}`);

    const labels = svg
      .append('g')
      .selectAll('text')
      .data(graphData.nodes)
      .enter()
      .append('text')
      .attr('dy', '.35em')
      .attr('text-anchor', 'middle')
      .text((d) => d.name.slice(0, 15) + (d.name.length > 15 ? '...' : ''));

    simulation.on('tick', () => {
      link
        .attr('x1', (d) => d.source.x)
        .attr('y1', (d) => d.source.y)
        .attr('x2', (d) => d.target.x)
        .attr('y2', (d) => d.target.y);

      node
        .attr('cx', (d) => d.x)
        .attr('cy', (d) => d.y);

      labels
        .attr('x', (d) => d.x)
        .attr('y', (d) => d.y + graphData.nodeTypes[d.type].radius + 10);
    });

    return () => simulation.stop();
  }, [graphData]);

  return <svg ref={svgRef} width="600" height="400" className="mx-auto"></svg>;
}

export default GraphVisualization;