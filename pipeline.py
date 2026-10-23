from agents import build_search_agent, build_scrap_agent, writer_chain, critic_chain

def run_research_pipeline(topic: str) -> dict:

    state = {}

    print("\n"+"="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_results = search_agent.invoke({
        "messages" : [("user", f"Find recent and relevant and detailed information on the topic: {topic}")]
    })

    state['search_results'] = search_results['messages'][-1].content

    print("\n Search result ", state['search_results'])


    print("\n"+"="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_scrap_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick all the relevant URLs and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results']}"
        )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content

    print("\nscraped content: \n", state['scraped_content'])

    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"Search Results:\n{state['search_results']}\n\n"
        f"Scraped Content:\n{state['scraped_content']}"
    )

    state['report'] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    print("\n Final Report\n",state['report'])

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state['critique'] = critic_chain.invoke({
        "report": state['report'][-1].content
    })

    print("\n Critique\n", state['critique'])

    return state

if __name__ == "__main__":
    topic = input(f"\n Enter a research topic: ")
    research_state = run_research_pipeline(topic)