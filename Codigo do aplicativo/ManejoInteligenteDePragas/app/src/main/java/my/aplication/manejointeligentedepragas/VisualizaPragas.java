package my.aplication.manejointeligentedepragas;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;

//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.design.widget.NavigationView;
import com.google.android.material.navigation.NavigationView;
//import android.support.v4.view.GravityCompat;
import androidx.core.view.GravityCompat;
//import android.support.v4.widget.DrawerLayout;
import androidx.drawerlayout.widget.DrawerLayout;
//import android.support.v7.app.ActionBarDrawerToggle;
import androidx.appcompat.app.ActionBarDrawerToggle;
//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;
//import android.support.v7.widget.Toolbar;
import androidx.appcompat.widget.Toolbar;

import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;

import com.example.manejointeligentedepragas.R;

import my.aplication.manejointeligentedepragas.VisualizaAdapter.VisuPragasAdapter;
import my.aplication.manejointeligentedepragas.model.PragaModel;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

public class VisualizaPragas extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{

    ArrayList<String> nomePragas = new ArrayList<>();
    ArrayList<Integer> codPragas = new ArrayList<>();
    private ArrayList<PragaModel> cards = new ArrayList<>();

    EditText edtPesquisaPragas;
    private ArrayList<String> pesquisa = new ArrayList<String>();
    private ArrayList<Integer> codPragapesquisa = new ArrayList<Integer>();
    Boolean pesquisado = false;

    Integer cod_Planta;

    private DrawerLayout drawerLayout;

    public String tipoUsu;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_visualiza_pragas);

        cod_Planta = getIntent().getIntExtra("Cod_Planta", 0);

        Utils u = new Utils();
        if(!u.isConected(getBaseContext())) {
            ExibeCaixaDialogo();
        }else{

        }

        final ListView listView = findViewById(R.id.ListViewPragas);
        edtPesquisaPragas = findViewById(R.id.PesquisaPragas);

        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_visuPraga);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_visuPraga);
        NavigationView navigationView = findViewById(R.id.nav_view_visuPraga);
        navigationView.setNavigationItemSelectedListener(this);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawerLayout.addDrawerListener(toggle);
        toggle.syncState();
        View headerView = navigationView.getHeaderView(0);

        Controller_Usuario controller_usuario = new Controller_Usuario(getBaseContext());
        String nomeUsu = controller_usuario.getUser().getNome();
        String emailUsu = controller_usuario.getUser().getEmail();

        TextView nomeMenu = headerView.findViewById(R.id.nomeMenu);
        nomeMenu.setText(nomeUsu);

        TextView emailMenu = headerView.findViewById(R.id.emailMenu);
        emailMenu.setText(emailUsu);


        setTitle("MIP² | Pragas");


        //esconder teclado
        this.getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_HIDDEN);

        if(cod_Planta != 0){
            ResgataPragasEspecificas(listView,cod_Planta);
        }else{
            ResgataPragas(listView);
        }


        //pesquisa
        edtPesquisaPragas.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {

            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                Pesquisar();
                listView.setAdapter(new ArrayAdapter<String>(VisualizaPragas.this, android.R.layout.simple_list_item_1, pesquisa));
                if(s==null){
                    pesquisado = false;
                }
            }

            @Override
            public void afterTextChanged(Editable s) {
                Pesquisar();
                listView.setAdapter(new ArrayAdapter<String>(VisualizaPragas.this, android.R.layout.simple_list_item_1, pesquisa));
                if(s==null){
                    pesquisado = false;
                }
            }
        });

        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {

                if(pesquisado ==true) {
                    Intent i = new Intent(VisualizaPragas.this, InfoPraga.class);
                    i.putExtra("Cod_Praga", codPragapesquisa.get(position));
                    startActivity(i);
                }else{
                    Intent i = new Intent(VisualizaPragas.this, InfoPraga.class);
                    i.putExtra("Cod_Praga", codPragas.get(position));
                    startActivity(i);
                }
            }
        });

    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            super.onBackPressed();
        }
    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
        switch (menuItem.getItemId()){
            case R.id.drawerPerfil:
                Controller_Usuario cu = new Controller_Usuario(getBaseContext());
                tipoUsu = cu.getUser().getTipo();
                if(tipoUsu == null){
                    Toast.makeText(VisualizaPragas.this,"Para acessar seu perfil, faça login!", Toast.LENGTH_LONG).show();
                }else{
                    Intent i= new Intent(this, Perfil.class);
                    startActivity(i);
                }
                break;
            case R.id.drawerProp:
                Controller_Usuario cu1 = new Controller_Usuario(getBaseContext());
                tipoUsu = cu1.getUser().getTipo();
                if(tipoUsu==null){
                    Toast.makeText(VisualizaPragas.this,"Para acessar as propriedades, faça login!", Toast.LENGTH_LONG).show();
                }else{
                    Intent prop= new Intent(this, Propriedades.class);
                    startActivity(prop);
                }
                break;

            case R.id.drawerPlantas:
                Intent j = new Intent(this, VisualizaPlantas.class);
                startActivity(j);
                break;

            case R.id.drawerPrag:
                Intent k = new Intent(this, VisualizaPragas.class);
                startActivity(k);
                break;

            case R.id.drawerMet:
                Intent l = new Intent(this, VisualizaMetodos.class);
                startActivity(l);
                break;

            case R.id.drawerSobreMip:
                Intent p = new Intent(this, SobreMIP.class);
                startActivity(p);
                break;

            case R.id.drawerTutorial:
                SharedPreferences pref = getApplicationContext().getSharedPreferences("myPrefs",MODE_PRIVATE);
                SharedPreferences.Editor editor = pref.edit();
                editor.putBoolean("isIntroOpened",false);
                editor.commit();

                Intent intro = new Intent(this, IntroActivity.class);
                startActivity(intro);
                break;

            case R.id.drawerSobre:
                Intent pp = new Intent(this, Sobre.class);
                startActivity(pp);
                break;

            case R.id.drawerReferencias:
                Intent pi = new Intent(this, Referencias.class);
                startActivity(pi);
                break;

            case R.id.drawerRecomendações:
                Intent pa = new Intent(this, RecomendacoesMAPA.class);
                startActivity(pa);
                break;
        }
        drawerLayout.closeDrawer(GravityCompat.START);
        return true;
    }


    public void ResgataPragas(final ListView listView){
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            //Toast.makeText(this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
        }else { // se tem acesso à internet
            String url = "https://mip.software/phpapp/visualizaPragas.php";

            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++){
                            JSONObject obj = array.getJSONObject(i);
                            nomePragas.add(obj.getString("NomePraga"));
                            codPragas.add(obj.getInt("Cod_Praga"));
                        }


                        ArrayAdapter<String> adapter =
                                new ArrayAdapter<String>(VisualizaPragas.this, android.R.layout.simple_list_item_1, nomePragas);
                        listView.setAdapter(adapter);

                    } catch (JSONException e) {
                        Toast.makeText(VisualizaPragas.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(VisualizaPragas.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));

        }
    }

    public void ResgataPragasEspecificas(final ListView listView, int codPlanta){
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            Toast.makeText(this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
        }else { // se tem acesso à internet
            String url = "https://mip.software/phpapp/selecionarPragasEspecificas.php?Cod_Planta=" + codPlanta;
            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    try {
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++){
                            JSONObject obj = array.getJSONObject(i);
                            nomePragas.add(obj.getString("Nome"));
                            codPragas.add(obj.getInt("Cod_Praga"));
                        }

                        ArrayAdapter<String> adapter =
                                new ArrayAdapter<String>(VisualizaPragas.this, android.R.layout.simple_list_item_1, nomePragas);
                        listView.setAdapter(adapter);


                    } catch (JSONException e) {
                        Toast.makeText(VisualizaPragas.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(VisualizaPragas.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));

        }
    }

    public void Pesquisar(){
        int textlength = edtPesquisaPragas.getText().length();
        pesquisa.clear();
        codPragapesquisa.clear();
        pesquisado = true;

        for (int i = 0; i < nomePragas.size(); i++ ) {
            if (textlength <= nomePragas.get(i).length()) {
                //if (edtPesquisaPragas.getText().toString().equalsIgnoreCase((String)nomePragas.get(i).subSequence(0, textlength))) {
                if(nomePragas.get(i).contains(edtPesquisaPragas.getText().toString())){
                    pesquisa.add(nomePragas.get(i));
                    codPragapesquisa.add(codPragas.get(i));
                }else if(edtPesquisaPragas.getText().toString().equalsIgnoreCase((String)nomePragas.get(i).subSequence(0, textlength))){
                    pesquisa.add(nomePragas.get(i));
                    codPragapesquisa.add(codPragas.get(i));
                }
            }
        }
    }

    public void ExibeCaixaDialogo() {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(this);
        dlgBox.setTitle("Aviso!");
        dlgBox.setMessage("Por enquanto, você só pode acessar as informações online! Esta função será disponibilizada no futuro!");
        dlgBox.setCancelable(false);
        dlgBox.setPositiveButton("Entendi", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                onBackPressed();
            }
        });
        dlgBox.show();
    }
}
