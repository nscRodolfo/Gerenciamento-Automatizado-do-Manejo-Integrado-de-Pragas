package my.aplication.manejointeligentedepragas.Crontroller;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;

import my.aplication.manejointeligentedepragas.Banco.Banco;
import my.aplication.manejointeligentedepragas.model.TalhaoModel;

import java.util.ArrayList;

public class Controller_Talhao {

    private SQLiteDatabase db;
    private Banco bancoLocal;

    public Controller_Talhao(Context context){
        bancoLocal = new Banco(context);
    }

    public boolean addTalhao(TalhaoModel talhaoModel){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Talhao, talhaoModel.getCod_Talhao());
        cv.put(bancoLocal.nomeTalhao, talhaoModel.getNome());
        cv.put(bancoLocal.Aplicado, talhaoModel.isAplicado());
        cv.put(bancoLocal.fk_Cultura_Cod_Cultura, talhaoModel.getFk_Cod_Cultura());
        cv.put(bancoLocal.fk_Planta_Cod_Planta_Talhao, talhaoModel.getFk_Cod_Planta());
        return db.insert(bancoLocal.Talhao, null, cv) != -1;
    }

    public boolean updateTalhao(TalhaoModel talhaoModel){
        db = bancoLocal.getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put(bancoLocal.Cod_Talhao, talhaoModel.getCod_Talhao());
        cv.put(bancoLocal.nomeTalhao, talhaoModel.getNome());
        cv.put(bancoLocal.Aplicado, talhaoModel.isAplicado());
        cv.put(bancoLocal.fk_Cultura_Cod_Cultura, talhaoModel.getFk_Cod_Cultura());
        cv.put(bancoLocal.fk_Planta_Cod_Planta_Talhao, talhaoModel.getFk_Cod_Planta());
        String where = " Cod_Talhao = '"+ talhaoModel.getCod_Talhao()+"'";
        return db.update(bancoLocal.Talhao, cv, where, null) > 0;
    }

    public ArrayList<TalhaoModel> getTalhao(int codCultura) {
        db = bancoLocal.getWritableDatabase();
        String query = "SELECT * FROM " + bancoLocal.Talhao+" WHERE fk_Cultura_Cod_Cultura=" +codCultura+" ORDER BY "+bancoLocal.nomeTalhao;
        ArrayList<TalhaoModel> talhoes = new ArrayList<>();
        Cursor c = db.rawQuery(query, null);
        if (c.moveToFirst()) {
            do {
                TalhaoModel talhaoModel = new TalhaoModel();
                talhaoModel.setCod_Talhao(c.getInt(0));
                talhaoModel.setNome(c.getString(1));
                boolean value = (c.getInt(2) == 1); // conferir
                talhaoModel.setAplicado(value);
                talhaoModel.setFk_Cod_Cultura(c.getInt(3));
                talhaoModel.setFk_Cod_Planta(c.getInt(4));
                talhoes.add(talhaoModel);
            } while (c.moveToNext());
        }
        c.close();
        return talhoes;
    }

    public void removerTalhao(){
        db = bancoLocal.getWritableDatabase();
        String query = "";
        db.delete(bancoLocal.Talhao, query, null);
    }

    public void removerTalhaoEspecifica(TalhaoModel talhaoModel){
        db = bancoLocal.getWritableDatabase();
        String where = " Cod_Talhao = '"+ talhaoModel.getCod_Talhao()+"'";
        db.delete(bancoLocal.Talhao, where, null);
    }
}
