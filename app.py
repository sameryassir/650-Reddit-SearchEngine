'''
Author: Prithvijit Dasgupta
Modified by: Zim Gong

This is the FastAPI start index. Currently it has 4 paths:

1. GET / -> Fetches the test bench HTML file. Used by browsers
2. POST /search -> This is the main search API responsible for performing the search across the index
3. GET /cache/:query/page/:page -> This path is meant to provide a cached response for pagination purposes
'''

# importing external modules
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from threading import Timer
import math

# importing internal modules
from models import QueryModel, APIResponse, PaginationModel
from pipeline import initialize

# Set up template rendering
templates = Jinja2Templates(directory="templates")

# Initialize the search engine
algorithm = initialize()

# Global variables for cache and timers
pagination_cache = {}
timer_mgr = {}

# Configuration constants
PAGE_SIZE = 10
CACHE_TIME = 3600  # Cache timeout in seconds

# FastAPI application
app = FastAPI()

# Cache deletion function to remove entries after a timeout
def delete_from_cache(query):
    if query in pagination_cache:
        del pagination_cache[query]
        del timer_mgr[query]

# API paths begin here

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    """Serve the search engine homepage."""
    return templates.TemplateResponse("index.html", {"request": request, "results": None})


@app.post('/search', response_class=HTMLResponse)
async def doSearch(request: Request, query: str = Form(...)):
    """Perform the search and render results."""
    response = algorithm.search(query)
    
    # Cache the results
    global pagination_cache, timer_mgr
    pagination_cache[query] = response
    pagination_cache[f'{query}_max_page'] = math.ceil(len(response) / PAGE_SIZE) - 1
    t = Timer(CACHE_TIME, delete_from_cache, [query])
    timer_mgr[query] = t
    t.start()

    # Render search results in HTML
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": response[:PAGE_SIZE],  # Display the first page
        },
    )


@app.get('/cache/{query}/page/{page}')
async def getCache(query: str, page: int) -> APIResponse:
    """Retrieve a cached page of results for a query."""
    if query not in pagination_cache:
        raise HTTPException(status_code=404, detail="Query not found in cache. Perform a search first.")

    max_page = pagination_cache.get(f'{query}_max_page', 0)
    page = max(0, min(page, max_page))  # Clamp page within valid range

    prev_page = max(0, page - 1)
    next_page = min(max_page, page + 1)

    results = pagination_cache[query][page * PAGE_SIZE:(page + 1) * PAGE_SIZE]

    return APIResponse(
        results=results,
        page=PaginationModel(
            prev=f'/cache/{query}/page/{prev_page}' if page > 0 else None,
            next=f'/cache/{query}/page/{next_page}' if page < max_page else None
        )
    )


@app.on_event('shutdown')
def timer_shutdown():
    """Ensure all timers are canceled during application shutdown."""
    for key, timer in timer_mgr.items():
        try:
            timer.cancel()
        except Exception as e:
            print(f"Error cancelling timer for {key}: {e}")

