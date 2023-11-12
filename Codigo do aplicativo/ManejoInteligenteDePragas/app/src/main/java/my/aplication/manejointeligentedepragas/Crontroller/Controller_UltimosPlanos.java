package my.aplication.manejointeligentedepragas.Crontroller;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;

import my.aplication.manejointeligentedepragas.Banco.Banco;

public class Controller_UltimosPlanos {

    private SQLiteDatabase db;
    private Banco bancoLocal;

    public Controller_UltimosPlanos(Context context){
        bancoLocal = new Banco(context);
    }

    public boolean addUltimosPlanos(String data, int fkTalhao, int fkPraga){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.fk_Praga_Cod_Praga, fkPraga);
        cv.put(bancoLocal.fk_Talhao_Cod_Talhao, fkTalhao);
        cv.put(bancoLocal.Data, data);
        return db.insert(bancoLocal.UltimosPlanos, null, cv) != -1;
    }

    public String getUltimosPlanos(int codTalhao, int codPraga) {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT UltimosPlanos.Data FROM " + bancoLocal.UltimosPlanos+" WHERE fk_Praga_Cod_Praga="+codPraga+" AND fk_Talhao_Cod_Talhao="+codTalhao+";";
        String ultimaData = "";
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                ultimaData = c.getString(0);

            } while (c.moveToNext());
        }
        c.close();
        return ultimaData;
    }

    public void removerUltimosPlanos(){
        db = bancoLocal.getWritableDatabase();
        String query = "";
        db.delete(bancoLocal.UltimosPlanos, query, null);
    }

}
