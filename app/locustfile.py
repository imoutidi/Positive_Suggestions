from locust import HttpUser, TaskSet, task, between

class UserBehavior(TaskSet):
    @task
    def suggest(self):
        self.client.post("/sentence_sentiment", data="the application was <blank>",
                         auth=("user_moutidis", "31415"))

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    # Specifies that the simulated user will wait between 1 and 5 seconds between each task.
    wait_time = between(2, 20)