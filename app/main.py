from app.sql_chain import sql_chain
from fastapi import FastAPI
from fastapi import Request
app = FastAPI()




@app.post("/sql_chain")
async def run_sql_chain_endpoint(request: Request):
    data = await request.json()
    input = data.get('input')
    output = sql_chain(input)
    return {"output": output}