from BetterTherapy.db.query import get_ready_requests

reqs = get_ready_requests(hours=1)

print(reqs)
