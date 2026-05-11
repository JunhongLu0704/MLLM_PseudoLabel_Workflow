from __future__ import annotations

from typing import Any, TypedDict

from .data import ImageRecord

HAS_LANGGRAPH = False
END = '__end__'
StateGraph = None  # type: ignore[assignment]


class ImageWorkflowState(TypedDict, total=False):
    record: ImageRecord
    merged_prior: dict[str, Any]
    prior_status: str
    duplicate_cluster_info: dict[str, Any]
    candidate_pool: dict[str, Any]
    rows: list[dict[str, Any]]
    image_summary: dict[str, Any]


def build_image_graph(runner: Any):
    global END, StateGraph, HAS_LANGGRAPH
    if not HAS_LANGGRAPH:
        try:
            from langgraph.graph import END as LANGGRAPH_END, StateGraph as LANGGRAPH_STATE_GRAPH
        except ImportError as exc:  # pragma: no cover - handled by caller/runtime
            raise RuntimeError('langgraph is required for the LangGraph workflow engine.') from exc
        END = LANGGRAPH_END
        StateGraph = LANGGRAPH_STATE_GRAPH
        HAS_LANGGRAPH = True
    graph = StateGraph(ImageWorkflowState)

    def build_priors(state: ImageWorkflowState) -> dict[str, Any]:
        record = state['record']
        merged_prior, prior_status = runner._build_priors(record)  # noqa: SLF001
        return {'merged_prior': merged_prior, 'prior_status': prior_status}

    def build_duplicate_clusters(state: ImageWorkflowState) -> dict[str, Any]:
        record = state['record']
        result = runner._build_duplicate_cluster_info(record)  # noqa: SLF001
        return {'duplicate_cluster_info': result}

    def build_candidates(state: ImageWorkflowState) -> dict[str, Any]:
        record = state['record']
        result = runner._build_candidate_pool(record, state.get('duplicate_cluster_info', {}))  # noqa: SLF001
        return {'candidate_pool': result}

    def process_proposals(state: ImageWorkflowState) -> dict[str, Any]:
        record = state['record']
        merged_prior = state['merged_prior']
        prior_status = state['prior_status']
        if runner.mode == 'mllm_reference_workflow_256':
            rows = runner._process_reference_workflow_proposals(  # noqa: SLF001
                record,
                merged_prior,
                prior_status,
                state.get('duplicate_cluster_info', {}),
                state.get('candidate_pool', {}),
            )
        else:
            rows = []
            for proposal in record.proposals:
                rows.append(runner._process_proposal(record, proposal, merged_prior, prior_status))  # noqa: SLF001
        return {'rows': rows}

    def finalize_image(state: ImageWorkflowState) -> dict[str, Any]:
        record = state['record']
        rows = state.get('rows', [])
        image_summary = {
            'image_id': record.image_id,
            'num_proposals': len(record.proposals),
            'num_valid_proposals': sum(1 for row in rows if not row['degenerate_bbox']),
            'mllm_prior_status': state.get('prior_status', 'not_called'),
            'duplicate_cluster_count': int(state.get('duplicate_cluster_info', {}).get('cluster_count', 0) or 0),
            'candidate_count': len(state.get('candidate_pool', {}).get('candidate_proposal_ids', [])),
        }
        runner.trace.image_trace(image_summary)
        return {'image_summary': image_summary}

    graph.add_node('build_priors', build_priors)
    graph.add_node('build_duplicate_clusters', build_duplicate_clusters)
    graph.add_node('build_candidates', build_candidates)
    graph.add_node('process_proposals', process_proposals)
    graph.add_node('finalize_image', finalize_image)
    graph.set_entry_point('build_priors')
    if runner.mode == 'mllm_reference_workflow_256':
        graph.add_edge('build_priors', 'build_duplicate_clusters')
        graph.add_edge('build_duplicate_clusters', 'build_candidates')
        graph.add_edge('build_candidates', 'process_proposals')
    else:
        graph.add_edge('build_priors', 'process_proposals')
    graph.add_edge('process_proposals', 'finalize_image')
    graph.add_edge('finalize_image', END)
    return graph.compile()
