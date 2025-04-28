from NeuMF.userInteractions import recommend_cars_from_recent_interactions
from app_db_context import AppDbContext

from app_db_context import AppDbContext
import pandas as pd

class UserRepository:
    @staticmethod
    def get_recent_interactions(user_id):
        with AppDbContext() as db:
            query = """
                SELECT cs."Make" AS brand, cs."Model" as model
                FROM "UserInteractions" ui
                JOIN "Cars" c ON ui."CarId" = c."Id"
                JOIN "CarSpecs" cs ON c."SpecId" = cs."Id"
                WHERE ui."UserId" = %s
                ORDER BY ui."InteractedAt" DESC
                LIMIT 20;

            """
            rows = db.execute(query, (user_id,))
            if not rows:
                recent_df = pd.DataFrame(columns=["brand", "model"])
            else:
                recent_df = pd.DataFrame(rows, columns=["brand", "model"])

            if len(recent_df) < 20:
                missing = 20 - len(recent_df)
                # Fetch random cars
                with AppDbContext() as db:
                    random_query = """
                               SELECT cs."Make" AS brand, cs."Model"
                               FROM "Cars" c
                               JOIN "CarSpecs" cs ON c."SpecId" = cs."Id"
                               ORDER BY RANDOM()
                               LIMIT %s;
                           """
                    random_rows = db.execute(random_query, (missing,))
                    if random_rows:
                        random_df = pd.DataFrame(random_rows, columns=["brand", "model"])
                        recent_df = pd.concat([recent_df, random_df], ignore_index=True)
            return recent_df

class InventoryRepository:
    @staticmethod
    def get_live_inventory():
        with AppDbContext() as db:
            query = """
                SELECT DISTINCT 
                    LOWER(REPLACE(cs."Make", ' ', '')) || '_' || LOWER(REPLACE(cs."Model", ' ', '')) AS item_id
                FROM "Auctions" a
                JOIN "Cars" c ON a."CarId" = c."Id"
                JOIN "CarSpecs" cs ON c."SpecId" = cs."Id"
                WHERE a."StartTime" >= NOW() - INTERVAL '7 days';
            """
            rows = db.execute(query)
            if not rows:
                return set()
            return set(row[0] for row in rows)
